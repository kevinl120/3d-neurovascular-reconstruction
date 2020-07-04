import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Conv2D, MaxPooling3D, UpSampling3D, Conv3D, Conv3DTranspose, Dropout
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.models import Model

from constants import *

class GAN():
    def __init__(self):
        self.generator = self.make_generator()
        self.discriminator = self.make_discriminator()
        self.discriminator.trainable = false

    def make_generator(self, **kwargs):
        inputs = Input(shape=(200))
        x = inputs
        x = Dense(32768, activation='relu')(x)
        x = Reshape((4, 4, 4, 512))(x)
        x = Conv3DTranspose(256, 4, 2, activation='relu', padding='same')(x)
        x = Conv3DTranspose(128, 4, 2, activation='relu', padding='same')(x)
        x = Conv3DTranspose(64, 4, 2, activation='relu', padding='same')(x)
        x = Conv3DTranspose(1, 4, 2, activation='relu', padding='same')(x)
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

def main():
    m = GAN()
    m.generator.summary()

if __name__ == '__main__':
    main()