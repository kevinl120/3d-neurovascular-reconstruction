import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Conv2D, MaxPooling2D, UpSampling3D, Conv3D, Conv3DTranspose, Dropout, Lambda, MaxPooling3D
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.models import Model

from constants import *
import preprocess

def weighted_bce(y_true, y_pred):
    # weights = (y_true * 2) + 1
    weights = 1
    # weights *= res
    bce = K.binary_crossentropy(y_true, y_pred)
    weighted_bce = K.mean(bce * weights)
    return weighted_bce

# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# then z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def make_vae():
    latent_dim = 128

    inputs = Input(shape=(VOXEL_SHAPE[0], VOXEL_SHAPE[1], VOXEL_SHAPE[2], 1))
    x = inputs

    # Encoder
    x = Conv3D(16, 3, activation='relu', padding='same')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = Dropout(0.2)(x)
    x = Conv3D(32, 3, activation='relu', padding='same')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = Dropout(0.2)(x)
    x = Conv3D(64, 3, activation='relu', padding='same')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = Dropout(0.2)(x)
    x = Conv3D(128, 3, activation='relu', padding='same')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = Dropout(0.2)(x)

    # shape info needed to build decoder model
    shape = K.int_shape(x)

    # generate latent vector Q(z|X)
    x = Flatten()(x)
    x = Dense(2048, activation='relu')(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(2048, activation='relu')(latent_inputs)
    x = Reshape((8, 8, 8, 4))(x)

    x = Dropout(0.2)(x)
    x = UpSampling3D(size=(2, 2, 2))(x)
    x = Dropout(0.2)(x)
    x = Conv3D(128, 3, activation='relu', padding='same')(x)
    x = Dropout(0.2)(x)
    x = UpSampling3D(size=(2, 2, 2))(x)
    x = Conv3D(64, 3, activation='relu', padding='same')(x)
    x = Dropout(0.2)(x)
    x = UpSampling3D(size=(2, 2, 2))(x)
    x = Conv3D(32, 3, activation='relu', padding='same')(x)
    x = Dropout(0.2)(x)
    outputs = Conv3D(1, 3, activation='sigmoid', padding='same')(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')

    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae')

    reconstruction_loss = weighted_bce(K.flatten(inputs), K.flatten(outputs))
    reconstruction_loss *= np.prod(VOXEL_SHAPE)
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)

    return vae

def main():
    x_train, y_train = preprocess.load_data()
    x_train, y_train = unison_shuffled_copies(x_train, y_train)

    vae = make_vae()
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    vae.compile(optimizer=opt)
    history = vae.fit(y_train, batch_size=4, epochs=20, validation_split=0.1)

    # enc.summary()
    # dec.summary()
    # vae.summary()

if __name__ == '__main__':
    main()