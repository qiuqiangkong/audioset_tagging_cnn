import tensorflow as tf
import numpy as np

from tf_stft import Spectrogram, Logmel


data_length = 320000
norm = None     # None | 'ortho'
np.random.seed(0)

# Spectrogram parameters
sample_rate = 32000
n_fft = 1024
hop_length = 320
win_length = 1024
window = 'hann'
center = True
dtype = np.complex64
pad_mode = 'reflect'

# Mel parameters
n_mels = 64
fmin = 50
fmax = 7000
ref = 1.0
amin = 1e-10
top_db = None

np_data = np.random.uniform(-1, 1, data_length)
 
def ConvBlock(out_channels, pool_size):
	return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=out_channels, kernel_size=(3, 3), strides=(1, 1), padding='same', 
            data_format='channels_first', activation='linear', 
            kernel_initializer='glorot_uniform', use_bias=False), 
        tf.keras.layers.BatchNormalization(axis=1), 
        tf.keras.layers.Activation('relu'), 

        tf.keras.layers.Conv2D(filters=out_channels, kernel_size=(3, 3), strides=(1, 1), padding='same', 
            data_format='channels_first', activation='linear', 
            kernel_initializer='glorot_uniform', use_bias=False), 
        tf.keras.layers.BatchNormalization(axis=1), 
        tf.keras.layers.Activation('relu'), 

        tf.keras.layers.AveragePooling2D(pool_size=pool_size, data_format='channels_first')        
    ])
	

def max_avg(x):
    _max = tf.math.reduce_max(x, axis=2)
    _avg = tf.math.reduce_mean(x, axis=2)
    return _max + _avg


'''
tf.keras.layers.BatchNormalization(axis=1), 
tf.keras.layers.Activation('relu'), 

tf.keras.layers.Conv2D(filters=out_channels, kernel_size=(3, 3), strides=(1, 1), padding='same', 
    data_format='channels_first', activation='linear', 
    kernel_initializer='glorot_uniform', use_bias=False), 
tf.keras.layers.BatchNormalization(axis=1), 
tf.keras.layers.Activation('relu'), 

tf.keras.layers.AveragePooling2D(pool_size=pool_size, data_format='channels_first')
'''

classes_num = 527

input = tf.keras.layers.Input((data_length,))
x = Spectrogram(win_length, hop_length)(input)
x = Logmel(sample_rate, win_length, n_mels, fmin, fmax)(x)
x = tf.keras.layers.Lambda(lambda x: tf.keras.backend.expand_dims(x, axis=1))(x)
x = tf.keras.layers.BatchNormalization(axis=3)(x)
x = ConvBlock(out_channels=64, pool_size=(2, 2))(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = ConvBlock(out_channels=128, pool_size=(2, 2))(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = ConvBlock(out_channels=256, pool_size=(2, 2))(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = ConvBlock(out_channels=512, pool_size=(2, 2))(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = ConvBlock(out_channels=1024, pool_size=(2, 2))(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = ConvBlock(out_channels=2048, pool_size=(1, 1))(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Lambda(lambda x: tf.math.reduce_mean(x, axis=3))(x)
x = tf.keras.layers.Lambda(lambda x: max_avg(x))(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(2048, activation='relu', kernel_initializer='glorot_uniform')(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(classes_num, activation='sigmoid')(x)

output = x

model = tf.keras.Model(inputs=[input], outputs=[output])
a1 = model.predict(np_data[None, :])

import crash
asdf(x)