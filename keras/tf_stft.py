import tensorflow as tf

import numpy as np
import librosa
import math


def Stft(win_length, hop_length):
    return tf.keras.Sequential([
        tf.keras.layers.Lambda(tf.pad, arguments={'paddings': [[0, 0], [win_length // 2, win_length // 2]], 'mode': 'REFLECT'}), 
        tf.keras.layers.Lambda(tf.signal.stft, arguments={'frame_length': win_length, 'frame_step': hop_length, 'window_fn': tf.signal.hann_window})
    ])


def Spectrogram(win_length, hop_length):
    return tf.keras.Sequential([
        # tf.keras.layers.Lambda(stft, arguments={'win_length': win_length, 'hop_length': hop_length}), 
        Stft(win_length, hop_length), 
        tf.keras.layers.Lambda(lambda x: tf.math.real(x) ** 2 + tf.math.imag(x) ** 2)
    ])


def log10(x):
    return tf.math.log(x) / tf.math.log(10.)

 
def Logmel(sample_rate, win_length, n_mels, fmin, fmax, amin=1e-10, top_db=80.0):
    melW = tf.constant(librosa.filters.mel(sr=sample_rate, n_fft=win_length, n_mels=n_mels, fmin=fmin, fmax=fmax).T, dtype=tf.float32)

    def _logmel(x):
        x = tf.keras.backend.dot(x, melW)
        x = tf.clip_by_value(x, amin, np.inf)
        x = 10. * log10(x)

        if top_db is not None:
            x = tf.clip_by_value(x, tf.reduce_max(x) - top_db, np.inf)
        
        return x

    return tf.keras.layers.Lambda(_logmel)


def debug(select):
    """Compare numpy + librosa and pytorch implementation result. For debug. 

    Args:
      select: 'dft' | 'logmel'
    """

    if select == 'stft':
        data_length = 32000
        np.random.seed(0)

        sample_rate = 16000
        n_fft = 1024
        hop_length = 250
        win_length = 1024
        window = 'hann'
        center = True
        dtype = np.complex64
        pad_mode = 'reflect'

        # Data
        np_data = np.random.uniform(-1, 1, data_length)

        # Numpy stft matrix
        np_stft_matrix = librosa.core.stft(y=np_data, n_fft=n_fft, 
            hop_length=hop_length, window=window, center=center).T

        input = tf.keras.layers.Input((data_length,))
        output = Stft(win_length, hop_length)(input)
        model = tf.keras.Model(inputs=[input], outputs=[output])
        
        tf_stft_matrix = model.predict(np_data[None, :])

        print('Comparing librosa and pytorch implementation of DFT. All numbers '
            'below should be close to 0.')
        print(np.mean(np.abs(np.real(np_stft_matrix) - np.real(tf_stft_matrix))))
        print(np.mean(np.abs(np.imag(np_stft_matrix) - np.imag(tf_stft_matrix))))
 
        '''
        # Numpy istft
        np_istft_s = librosa.core.istft(stft_matrix=np_stft_matrix.T, 
            hop_length=hop_length, window=window, center=center, length=data_length)

        , 'window_fn': tf.signal.inverse_stft_window_fn(hop_length, forward_window_fn=tf.signal.hann_window

        win = tf.signal.inverse_stft_window_fn(hop_length)
        input = tf.keras.layers.Input((data_length,))
        x = tf.keras.layers.Lambda(tf.pad, arguments={'paddings': [[0, 0], [0, 0]], 'mode': 'REFLECT'})(input)
        x = tf.keras.layers.Lambda(tf.signal.stft, arguments={'frame_length': win_length, 'frame_step': hop_length, 'window_fn': tf.signal.hann_window})(x)
        x = tf.keras.layers.Lambda(tf.signal.inverse_stft, arguments={'frame_length': win_length, 'frame_step': hop_length, 'window_fn': win})(x)
        # x = tf.keras.layers.Lambda(lambda x: x[:, 512 : -512])(x)
        output = x
        model = tf.keras.Model(inputs=[input], outputs=[output])
        tf_istft_s = model.predict(np_data[None, :])
        
        print(np.mean(np.abs(np_istft_s - tf_istft_s)))
        print(np.mean(np.abs(np_data - tf_istft_s)))
        '''

    elif select == 'logmel':

        data_length = 32000
        norm = None     # None | 'ortho'
        np.random.seed(0)

        # Spectrogram parameters
        sample_rate = 16000
        n_fft = 1024
        hop_length = 250
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

        # Data
        np_data = np.random.uniform(-1, 1, data_length)

        print('Comparing librosa and pytorch implementation of logmel '
            'spectrogram. All numbers below should be close to 0.')

        # Numpy log mel spectrogram
        np_stft_matrix = librosa.core.stft(y=np_data, n_fft=n_fft, hop_length=hop_length, 
            win_length=win_length, window=window, center=center, dtype=dtype, 
            pad_mode=pad_mode)

        np_pad = np.pad(np_data, int(n_fft // 2), mode=pad_mode)

        np_melW = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels,
            fmin=fmin, fmax=fmax).T

        np_mel_spectrogram = np.dot(np.abs(np_stft_matrix.T) ** 2, np_melW)

        np_logmel_spectrogram = librosa.core.power_to_db(
            np_mel_spectrogram, ref=ref, amin=amin, top_db=top_db)

        # TF mel spectrogram
        input = tf.keras.layers.Input((data_length,))
        x = Spectrogram(win_length, hop_length)(input)
        output = Logmel(sample_rate, win_length, n_mels, fmin, fmax)(x)
        
        model = tf.keras.Model(inputs=[input], outputs=[output])
        tf_logmel_spectrogram = model.predict(np_data[None, :])

        # Compare
        print(np.mean(np.abs(np_logmel_spectrogram - tf_logmel_spectrogram)))


if __name__ == '__main__':

    # Uncomment for debug
    if True:
        debug('stft')
        debug(select='logmel')