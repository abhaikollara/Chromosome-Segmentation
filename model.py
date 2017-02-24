import keras
from keras.models import Model
from keras.layers import Activation, BatchNormalization, Dense, Input
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras import backend as K

k_size=3

def custom_loss(y_true, y_pred):
    y_t = K.reshape(y_true,[-1,1])
    y_p = K.reshape(y_pred,[-1,4])
    losses = K.sparse_categorical_crossentropy(y_p,y_t, from_logits=True)
    return K.sum(losses)

def SegNet():
    inputs = Input([96,96,1])

    # Encoder
    e = Convolution2D(32,k_size,k_size,border_mode='same')(inputs)
    e = BatchNormalization()(e)
    e = Activation('relu')(e)
    e = MaxPooling2D()(e)

    e = Convolution2D(32,k_size,k_size,border_mode='same')(e)
    e = BatchNormalization()(e)
    e = Activation('relu')(e)
    e = MaxPooling2D()(e)

    e = Convolution2D(32,k_size,k_size,border_mode='same')(e)
    e = BatchNormalization()(e)
    e = Activation('relu')(e)
    e = MaxPooling2D()(e)

    # Decoder
    d = UpSampling2D()(e)
    d = BatchNormalization()(d)
    d = Convolution2D(32,k_size,k_size,border_mode='same')(d)
    d = Activation('relu')(d)

    d = UpSampling2D()(d)
    d = BatchNormalization()(d)
    d = Convolution2D(32,k_size,k_size,border_mode='same')(d)
    d = Activation('relu')(d)

    d = UpSampling2D()(d)
    d = BatchNormalization()(d)
    d = Convolution2D(32,k_size,k_size,border_mode='same')(d)
    d = Activation('relu')(d)

    out = Convolution2D(4,1,1)(d)

    model = Model(inputs, out)
    model.compile(optimizer='adam', loss=custom_loss)

    return model