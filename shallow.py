import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Conv2D, MaxPooling2D, UpSampling3D, Conv3D, Conv3DTranspose
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.models import Model

from constants import *
import preprocess


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def make_dnn(**kwargs):
    inputs = Input(shape=(PROJ_SHAPE[0], PROJ_SHAPE[1], NUM_VIEWS))
    x = inputs

    # Encoder
    x = Conv2D(32, 7, activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(32, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(32, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(32, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(32, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(32, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)

    # FC
    x = Dense(64, activation='relu')(x)
    x = Reshape((4, 4, 4, 1))(x)

    # Decoder
    x = UpSampling3D(size=(2, 2, 2))(x)
    x = Conv3D(32, 3, activation='relu', padding='same')(x)
    x = UpSampling3D(size=(2, 2, 2))(x)
    x = Conv3D(32, 3, activation='relu', padding='same')(x)
    x = UpSampling3D(size=(2, 2, 2))(x)
    x = Conv3D(32, 3, activation='relu', padding='same')(x)
    x = UpSampling3D(size=(2, 2, 2))(x)
    x = Conv3D(1, 3, activation='sigmoid', padding='same')(x)
    
    outputs = Reshape((64, 64, 64))(x)

    dnn = Model(inputs=inputs, outputs=outputs)
    return dnn


def main():
    x_train, y_train = preprocess.load_data()
    x_train, y_train = unison_shuffled_copies(x_train, y_train)
    dnn = make_dnn()
    dnn.compile(optimizer='Adam', loss=binary_crossentropy)
    dnn.summary()
    history = dnn.fit(x_train, y_train, batch_size=None, epochs=5, validation_split=0.1)

if __name__ == '__main__':
    main()