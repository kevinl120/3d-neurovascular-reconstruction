import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Conv2D, MaxPooling2D, UpSampling3D, Conv3D, Conv3DTranspose
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.models import Model

from constants import *
import preprocess


def weighted_bce(y_true, y_pred):
    weights = (y_true * 40) + 1.
    bce = K.binary_crossentropy(y_true, y_pred)
    weighted_bce = K.mean(bce * weights)
    return weighted_bce


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def make_dnn(**kwargs):
    inputs = Input(shape=(PROJ_SHAPE[0], PROJ_SHAPE[1], NUM_VIEWS))
    x = inputs

    # Encoder
    x = Conv2D(16, 7, activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(32, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(32, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)

    # FC
    x = Dense(2048, activation='relu')(x)
    x = Reshape((8, 8, 8, 4))(x)

    # Decoder
    x = UpSampling3D(size=(2, 2, 2))(x)
    x = Conv3D(128, 3, activation='relu', padding='same')(x)
    x = UpSampling3D(size=(2, 2, 2))(x)
    x = Conv3D(64, 3, activation='relu', padding='same')(x)
    x = UpSampling3D(size=(2, 2, 2))(x)
    x = Conv3D(32, 3, activation='relu', padding='same')(x)
    x = Conv3D(1, 3, activation='sigmoid', padding='same')(x)
    
    outputs = Reshape((64, 64, 64))(x)

    dnn = Model(inputs=inputs, outputs=outputs)
    return dnn


def main():
    x_train, y_train = preprocess.load_data()
    print(np.sum(x_train[0]))
    x_train, y_train = unison_shuffled_copies(x_train, y_train)
    dnn = make_dnn()
    dnn.compile(optimizer='Adam', loss=weighted_bce)
    dnn.summary()
    history = dnn.fit(x_train, y_train, batch_size=None, epochs=1, validation_split=0.1)

if __name__ == '__main__':
    main()