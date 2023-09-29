# MIT License
#
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import tensorflow as tf


def Fourier_filter(x, threshold=1, scale=0.9):
    x_dtype = x.dtype
    x = tf.cast(x, tf.float32)
    # FFT
    x_freq = tf.signal.fft3d(tf.cast(x, dtype=tf.complex64))
    x_freq = tf.signal.fftshift(x_freq, axes=(1, 2, 3))
    B, H, W, C = x_freq.get_shape().as_list()
    mask = np.ones((1, H, W, C), dtype=np.complex64)

    crow, ccol = H // 2, W // 2
    mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold, :] = scale
    x_freq = x_freq * mask

    # IFFT
    x_freq = tf.signal.ifftshift(x_freq, axes=(1, 2, 3))
    x_filtered = tf.signal.ifft3d(x_freq)
    x_filtered = tf.math.real(x_filtered)
    return tf.cast(x_filtered, x_dtype)


def free_u(h, hs_, active=False, b1=1.2, b2=1.4, s1=0.9, s2=0.2, axis=-1):
    if active:
        if h.get_shape().as_list()[axis] == 1280:
            h1, h2 = tf.split(h, num_or_size_splits=2, axis=axis)
            h = tf.keras.layers.Concatenate(axis=axis)([h1 * b1, h2])
            hs_ = Fourier_filter(hs_, threshold=1, scale=s1)
        if h.get_shape().as_list()[axis] == 640:
            h1, h2 = tf.split(h, num_or_size_splits=2, axis=axis)
            h = tf.keras.layers.Concatenate(axis=axis)([h1 * b2, h2])
            hs_ = Fourier_filter(hs_, threshold=1, scale=s2)
    return tf.keras.layers.Concatenate(axis=axis)([h, hs_])
