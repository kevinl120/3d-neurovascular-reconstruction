import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Conv2D, MaxPooling2D, UpSampling3D, Conv3D, Conv3DTranspose, Dropout
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.models import Model

from constants import *
import preprocess


arr = np.zeros((64,64,64))
for i in range(64):
    for j in range(64):
        for k in range(64):
            arr[i][j][k] = i+j+k

res = arr
res = np.minimum(res, np.flip(arr, axis=0))
res = np.minimum(res, np.flip(arr, axis=1))
res = np.minimum(res, np.flip(arr, axis=2))
res = np.minimum(res, np.flip(arr, axis=(0,1)))
res = np.minimum(res, np.flip(arr, axis=(0,2)))
res = np.minimum(res, np.flip(arr, axis=(1,2)))

def weighted_bce(y_true, y_pred):
    weights = (y_true * 2) + 1.
    weights *= res
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
    x = Conv2D(32, 7, activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)
    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)
    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)
    x = Conv2D(256, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)

    # FC
    x = Dense(4320, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Reshape((6, 6, 6, 20))(x)

    # Decoder
    x = Dropout(0.2)(x)
    x = UpSampling3D(size=(2, 2, 2))(x)
    x = Dropout(0.2)(x)
    x = Conv3DTranspose(256, 3, activation='relu', padding='valid')(x)
    x = Dropout(0.2)(x)
    x = UpSampling3D(size=(2, 2, 2))(x)
    x = Conv3DTranspose(128, 3, activation='relu', padding='valid')(x)
    x = Dropout(0.2)(x)
    x = UpSampling3D(size=(2, 2, 2))(x)
    x = Conv3DTranspose(64, 3, activation='relu', padding='valid')(x)
    x = Dropout(0.2)(x)
    x = Conv3DTranspose(1, 3, activation='sigmoid', padding='valid')(x)
    
    outputs = Reshape((64, 64, 64))(x)

    dnn = Model(inputs=inputs, outputs=outputs)
    return dnn


def main():
    # x_train, y_train = preprocess.load_data()
    # x_train, y_train = unison_shuffled_copies(x_train, y_train)
    dnn = make_dnn()
    # dnn.compile(optimizer='Adam', loss=weighted_bce)
    dnn.summary()
    # history = dnn.fit(x_train, y_train, batch_size=None, epochs=1, validation_split=0.1)

if __name__ == '__main__':
    main()