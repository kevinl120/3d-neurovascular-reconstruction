import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Conv2D, MaxPooling3D, UpSampling3D, Conv3D, Conv3DTranspose, Dropout
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from constants import *
import preprocess

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

    def make_generator(self, **kwargs):
        inputs = Input(shape=(self.latent_dim,))
        x = inputs
        x = Dense(32768, activation='relu')(x)
        x = Reshape((4, 4, 4, 512))(x)
        x = Conv3DTranspose(256, 4, 2, activation='relu', padding='same')(x)
        x = Conv3DTranspose(128, 4, 2, activation='relu', padding='same')(x)
        x = Conv3DTranspose(64, 4, 2, activation='relu', padding='same')(x)
        x = Conv3DTranspose(1, 4, 2, activation='sigmoid', padding='same')(x)
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
    
    def train(self, X_train, epochs=100, batch_size=32):
        d_loss = (999999, 0)
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        for epoch in range(epochs):
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)


            if d_loss[1] > 0.8:
                print('Discriminator accuracy last batch greater than 0.8, skipping discriminator training this batch.')
                # import pdb
                # pdb.set_trace()
                d_pred_real = np.round(self.discriminator.predict(imgs))
                d_pred_fake = np.round(self.discriminator.predict(gen_imgs))
                bce = tf.keras.losses.BinaryCrossentropy()
                d_loss_real = [bce(valid, d_pred_real).numpy(), np.mean(np.equal(valid, d_pred_real))]
                d_loss_fake = [bce(fake, d_pred_fake).numpy(), np.mean(np.equal(fake, d_pred_fake))]
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            else:
                d_loss_real = self.discriminator.train_on_batch(imgs, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.combined.train_on_batch(noise, valid)

            print("{} [D loss: {}, acc.: {:.2f}%] [G loss: {}]".format(epoch, d_loss[0], 100*d_loss[1], g_loss))

            if epoch % 10 == 0:
                self.sample_images(epoch)
    
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