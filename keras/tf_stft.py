import matplotlib.pyplot as plt
import tensorflow as tf

import numpy as np
import librosa
import math



def dft_matrix(n):
    (x, y) = np.meshgrid(np.arange(n), np.arange(n))
    omega = np.exp(-2 * np.pi * 1j / n)
    W = np.power(omega, x * y)
    return W

def idft_matrix(n):
    (x, y) = np.meshgrid(np.arange(n), np.arange(n))
    omega = np.exp(2 * np.pi * 1j / n)
    W = np.power(omega, x * y)
    return W


'''
def Stft(win_length, hop_length):
    return tf.keras.Sequential([
        tf.keras.layers.Lambda(tf.pad, arguments={'paddings': [[0, 0], [win_length // 2, win_length // 2]], 'mode': 'REFLECT'}), 
        tf.keras.layers.Lambda(tf.signal.stft, arguments={'frame_length': win_length, 'frame_step': hop_length, 'window_fn': tf.signal.hann_window})
    ])
'''

class Stft(object):
    def __init__(self, n_fft=2048, hop_length=None, win_length=None, 
        window='hann', center=True, pad_mode='reflect', freeze_parameters=True):
        """Implementation of STFT with Conv1d. The function has the same output 
        of librosa.core.stft
        """
        assert pad_mode in ['reflect']

        self.n_fft = n_fft
        self.center = center
        self.pad_mode = pad_mode
        self.trainable = not freeze_parameters

        # By default, use the entire frame
        if win_length is None:
            win_length = n_fft
        self.win_length = win_length

        # Set the default hop, if it's not already specified
        if hop_length is None:
            hop_length = int(win_length // 4)
        self.hop_length = hop_length

        fft_window = librosa.filters.get_window(window, win_length, fftbins=True)

        # Pad the window out to n_fft size
        self.fft_window = librosa.util.pad_center(fft_window, n_fft)


        # DFT & IDFT matrix
        self.W = dft_matrix(n_fft)

        self.out_channels = n_fft // 2 + 1

    def __call__(self, x):
        x = tf.keras.layers.Lambda(tf.pad, arguments={
            'paddings': [[0, 0], [self.win_length // 2, self.win_length // 2]], 'mode': 'REFLECT'})(x)
        x = tf.keras.layers.Lambda(tf.keras.backend.expand_dims, arguments={'axis': 1})(x)
        
        real_layer = tf.keras.layers.Conv1D(filters=self.out_channels, 
            kernel_size=self.n_fft, strides=self.hop_length, padding='valid', 
            data_format='channels_first', dilation_rate=1, use_bias=False, 
            trainable=self.trainable)
        real = real_layer(x)
        real_layer.set_weights([np.real(self.W[:, 0 : self.out_channels] * self.fft_window[:, None])[:, None, :]])

        imag_layer = tf.keras.layers.Conv1D(filters=self.out_channels, 
            kernel_size=self.n_fft, strides=self.hop_length, padding='valid', 
            data_format='channels_first', dilation_rate=1, use_bias=False, 
            trainable=self.trainable)
        imag = imag_layer(x)
        imag_layer.set_weights([np.imag(self.W[:, 0 : self.out_channels] * self.fft_window[:, None])[:, None, :]])

        real = tf.keras.layers.Permute(dims=(2, 1))(real)
        imag = tf.keras.layers.Permute(dims=(2, 1))(imag)
        real = tf.keras.layers.Lambda(tf.keras.backend.expand_dims, arguments={'axis': 1})(real)
        imag = tf.keras.layers.Lambda(tf.keras.backend.expand_dims, arguments={'axis': 1})(imag)
        # (batch_size, 1, time_steps, n_fft // 2 + 1)

        return real, imag


class ISTFT(object):
    """Unfinished. TF with ISTFT is not easy to implement.
    """
    def __init__(self, n_fft=2048, hop_length=None, win_length=None, 
        window='hann', center=True, pad_mode='reflect', freeze_parameters=True):
        """Implementation of ISTFT with Conv1d. The function has the same output 
        of librosa.core.istft
        """
        assert pad_mode in ['reflect']

        self.n_fft = n_fft
        self.center = center
        self.pad_mode = pad_mode
        self.trainable = not freeze_parameters

        # By default, use the entire frame
        if win_length is None:
            win_length = n_fft
        self.win_length = win_length

        # Set the default hop, if it's not already specified
        if hop_length is None:
            hop_length = int(win_length // 4)
        self.hop_length = hop_length

        ifft_window = librosa.filters.get_window(window, win_length, fftbins=True)

        # Pad the window out to n_fft size
        self.ifft_window = librosa.util.pad_center(ifft_window, n_fft)

        # DFT & IDFT matrix
        self.W = dft_matrix(n_fft) / n_fft

        self.out_channels = n_fft

    def __call__(self, real_stft, imag_stft, length=None):
        
        real_stft = tf.keras.layers.Lambda(lambda x: x[:, 0, :, :])(real_stft)
        imag_stft = tf.keras.layers.Lambda(lambda x: x[:, 0, :, :])(imag_stft)
        real_stft = tf.keras.layers.Permute(dims=(2, 1))(real_stft)
        imag_stft = tf.keras.layers.Permute(dims=(2, 1))(imag_stft)
        """(batch_size, freq_bins, time_steps)"""

        def _get_full_stft(x):
            return tf.concat((x, x[:, -2:0:-1, :]), axis=1)

        full_real_stft = tf.keras.layers.Lambda(_get_full_stft)(real_stft)
        full_imag_stft = tf.keras.layers.Lambda(_get_full_stft)(imag_stft)

        real_layer = tf.keras.layers.Conv1D(filters=self.out_channels, 
            kernel_size=1, strides=1, padding='valid', 
            data_format='channels_first', dilation_rate=1, use_bias=False, 
            trainable=self.trainable)
        real = real_layer(full_real_stft)

        real_layer.set_weights([np.real(self.W * self.ifft_window[None, :])[None, :, :]])

        imag_layer = tf.keras.layers.Conv1D(filters=self.out_channels, 
            kernel_size=1, strides=1, padding='valid', 
            data_format='channels_first', dilation_rate=1, use_bias=False, 
            trainable=self.trainable)
        imag = imag_layer(full_imag_stft)
        imag_layer.set_weights([np.imag(self.W * self.ifft_window[:, None])[None, :, :]])

        s_real = real - imag

        # Reserve space for reconstructed waveform
        if length:
            if self.center:
                padded_length = length + int(self.n_fft)
            else:
                padded_length = length
            n_frames = min(
                real_stft.shape[2], int(np.ceil(padded_length / self.hop_length)))
        else:
            n_frames = real_stft.shape[2]

        expected_signal_len = self.n_fft + self.hop_length * (n_frames - 1)
        expected_signal_len = self.n_fft + self.hop_length * (n_frames - 1)

        import crash
        asdf

        def _add(x):
            y = tf.zeros_like(tf.placeholder(tf.float32, shape=(x.shape[0], length)))

            # Overlap add
            for i in range(n_frames):
                y[:, i * self.hop_length : i * self.hop_length + self.n_fft] += s_real[:, :, i]

            return y

        y = tf.keras.layers.Lambda(_add)(s_real)

        '''
        ifft_window_sum = librosa.filters.window_sumsquare(self.window, n_frames,
            win_length=self.win_length, n_fft=self.n_fft, hop_length=self.hop_length)

        approx_nonzero_indices = np.where(ifft_window_sum > librosa.util.tiny(ifft_window_sum))[0]
        approx_nonzero_indices = torch.LongTensor(approx_nonzero_indices).to(device)
        ifft_window_sum = torch.Tensor(ifft_window_sum).to(device)
        
        y[:, approx_nonzero_indices] /= ifft_window_sum[approx_nonzero_indices][None, :]
        '''
        import crash
        asdf


class Spectrogram(object):
    def __init__(self, n_fft=2048, hop_length=None, win_length=None, 
        window='hann', center=True, pad_mode='reflect', power=2.0, 
        freeze_parameters=True):
        """Spectrogram.
        """

    def __call__(self, x):
        



'''
def Spectrogram(win_length, hop_length):
    return tf.keras.Sequential([
        # tf.keras.layers.Lambda(stft, arguments={'win_length': win_length, 'hop_length': hop_length}), 
        Stft(win_length, hop_length), 
        tf.keras.layers.Lambda(lambda x: tf.math.real(x) ** 2 + tf.math.imag(x) ** 2)
    ])
'''

class Spectrogram(object):
    def __init__(self, n_fft=2048, hop_length=None, win_length=None, 
        window='hann', center=True, pad_mode='reflect', power=2.0, 
        freeze_parameters=True):

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center
        self.pad_mode = pad_mode
        self.power = power
        self.freeze_parameters = freeze_parameters

    def __call__(self, x):
        
        (real, imag) = Stft(n_fft=n_fft, hop_length=hop_length, 
            win_length=win_length, window=window, center=center, 
            pad_mode=pad_mode, freeze_parameters=freeze_parameters)(x)

        spectrogram = tf.keras.layers.Lambda(
            lambda x, y: tf.math.pow(tf.math.real(x) ** 2 + tf.math.imag(y) ** 2, self.power / 2))(real, imag)

        return spectrogram


def log10(x):
    return tf.math.log(x) / tf.math.log(10.)

 
def LogmelFilterBank(sample_rate, win_length, n_mels, fmin, fmax, amin=1e-10, top_db=80.0):
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

        # TF stft matrix
        input = tf.keras.layers.Input((data_length,))
        (output_real, output_imag) = Stft(n_fft=n_fft, hop_length=hop_length, 
            win_length=win_length, window=window, center=center, 
            pad_mode=pad_mode, freeze_parameters=True)(input)
        model = tf.keras.Model(inputs=[input], outputs=[output_real, output_imag])
        (tf_real, tf_imag) = model.predict(np_data[None, :])

        print('Comparing librosa and pytorch implementation of DFT. All numbers '
            'below should be close to 0.')
        print(np.mean(np.abs(np.real(np_stft_matrix) - tf_real)))
        print(np.mean(np.abs(np.imag(np_stft_matrix) - tf_imag)))

        
        # Numpy istft
        """ Not done yet.
        np_istft_s = librosa.core.istft(stft_matrix=np_stft_matrix.T, 
            hop_length=hop_length, window=window, center=center, length=data_length)

        input_shape = (1, tf_real.shape[2], tf_real.shape[3])
        input_real = tf.keras.layers.Input(input_shape)
        input_imag = tf.keras.layers.Input(input_shape)
        tf_istft_s = ISTFT(n_fft=n_fft, hop_length=hop_length, 
            win_length=win_length, window=window, center=center, 
            pad_mode=pad_mode, freeze_parameters=True)(input_real, input_imag, data_length)
        """


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
        (output_real, output_imag) = Stft(n_fft=n_fft, hop_length=hop_length, 
            win_length=win_length, window=window, center=center, 
            pad_mode=pad_mode, freeze_parameters=True)(input)

        model = tf.keras.Model(inputs=[input], outputs=[output_real, output_imag])
        (tf_real, tf_imag) = model.predict(np_data[None, :])
        """
        input = tf.keras.layers.Input((data_length,))
        x = Spectrogram(win_length, hop_length)(input)
        output = Logmel(sample_rate, win_length, n_mels, fmin, fmax)(x)
        
        model = tf.keras.Model(inputs=[input], outputs=[output])
        tf_logmel_spectrogram = model.predict(np_data[None, :])

        # Compare
        print(np.mean(np.abs(np_logmel_spectrogram - tf_logmel_spectrogram)))
        """

if __name__ == '__main__':

    # Uncomment for debug
    if True:
        debug('stft')
        debug(select='logmel')