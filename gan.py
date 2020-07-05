from constants import *
import preprocess

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Conv2D, MaxPooling3D, UpSampling3D, Conv3D, Conv3DTranspose, Dropout
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def round_through(x):
    rounded = K.round(x)
    return x + K.stop_gradient(rounded - x)

def _hard_sigmoid(x):
    x = (0.5 * x) + 0.5
    return K.clip(x, 0, 1)

def binary_sigmoid(x):
    return round_through(_hard_sigmoid(x))


class GAN():
    def __init__(self):
        self.latent_dim = 200
        
        optimizer = Adam(0.0002, 0.5)

        self.generator = self.make_generator()
        self.discriminator = self.make_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)
        self.discriminator.trainable = False
        classification = self.discriminator(img)
        self.combined = Model(z, classification)
        self.combined.compile(optimizer=optimizer, loss='binary_crossentropy')


        self.epochs = 0
        self.skipped = 0
        self.d_loss = (9999999, 0)
        self.g_loss = 0

    def make_generator(self, **kwargs):
        inputs = Input(shape=(self.latent_dim,))
        x = inputs
        x = Dense(32768, activation='relu')(x)
        x = Reshape((4, 4, 4, 512))(x)
        x = Conv3DTranspose(256, 4, 2, activation='relu', padding='same')(x)
        x = Conv3DTranspose(128, 4, 2, activation='relu', padding='same')(x)
        x = Conv3DTranspose(64, 4, 2, activation='relu', padding='same')(x)
        x = Conv3DTranspose(1, 4, 2, activation=binary_sigmoid, padding='same')(x)
        outputs = x
        return Model(inputs, outputs)

    def make_discriminator(self, **kwargs):
        inputs = Input(shape=(VOXEL_SHAPE[0], VOXEL_SHAPE[1], VOXEL_SHAPE[2], 1))
        x = inputs
        x = Conv3D(64, 4, 2, activation='relu', padding='same')(x)
        x = Conv3D(128, 4, 2, activation='relu', padding='same')(x)
        x = Conv3D(256, 4, 2, activation='relu', padding='same')(x)
        x = Conv3D(512, 4, 2, activation='relu', padding='same')(x)
        x = Flatten()(x)
        outputs = Dense(1, activation='sigmoid')(x)
        return Model(inputs, outputs)
    
    def train(self, X_train, epochs=500, batch_size=4):
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for _ in range(epochs):
            self.epochs += 1
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            imgs = np.reshape(imgs, (-1, VOXEL_SHAPE[0], VOXEL_SHAPE[1], VOXEL_SHAPE[2], 1))
            gen_imgs = np.reshape(gen_imgs, (-1, VOXEL_SHAPE[0], VOXEL_SHAPE[1], VOXEL_SHAPE[2], 1))

            if self.d_loss[1] > 0.8:
                print('Discriminator accuracy last batch greater than 0.8, skipping discriminator training this batch. ({} in a row)'.format(self.skipped))
                self.skipped += 1
                d_pred_real = np.round(self.discriminator.predict(imgs))
                d_pred_fake = np.round(self.discriminator.predict(gen_imgs))
                bce = tf.keras.losses.BinaryCrossentropy()
                d_loss_real = [bce(valid, d_pred_real).numpy(), np.mean(np.equal(valid, d_pred_real))]
                d_loss_fake = [bce(fake, d_pred_fake).numpy(), np.mean(np.equal(fake, d_pred_fake))]
                self.d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            else:
                self.skipped = 1
                d_loss_real = self.discriminator.train_on_batch(imgs, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                self.d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            self.g_loss = self.combined.train_on_batch(noise, valid)

            print("{} [D loss: {}, acc.: {:.2f}%] [G loss: {}]".format(self.epochs, self.d_loss[0], 100*self.d_loss[1], self.g_loss))

            if self.epochs % 50 == 0:
                self.sample_images(self.epochs)
    
    def sample_images(self, epoch):
        r, c = 1, 1
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        np.save('gan_generation/{}.npy'.format(epoch), gen_imgs)


def main():
    gan = GAN()
    x_train, y_train = preprocess.load_data()
    gan.train(y_train)


if __name__ == '__main__':
    main()