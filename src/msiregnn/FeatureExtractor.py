""" Provides class definitions for FeatureExtractor layers for *Registration model."""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Resizing

__all__ = [
    "FeatureExtractor"
]

class FeatureExtractor(tf.keras.layers.Layer):
    def __init__(self, target_shape):
        super(FeatureExtractor, self).__init__()
        self.target_shape = target_shape
        self.layers_list = []

    def build(self, input_shape):5632
    current_shape = input_shape[1:3]
        target_shape = self.target_shape[1:3]

        while current_shape[0] > target_shape[0] or current_shape[1] > target_shape[1]:
            if current_shape[0] > target_shape[0]:
                self.layers_list.append(Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu'))
            if current_shape[1] > target_shape[1]:
                self.layers_list.append(Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu'))
            self.layers_list.append(MaxPooling2D(pool_size=2, strides=2, padding='same'))
            current_shape = (current_shape[0] // 2, current_shape[1] // 2)

        self.layers_list.append(Resizing(height=target_shape[0], width=target_shape[1]))
        self.layers_list.append(Conv2D(filters=self.target_shape[-1], kernel_size=1, activation='linear'))

    def call(self, inputs):
        x = inputs
        for layer in self.layers_list:
            x = layer(x)
            # print(f"Shape after {layer.__class__.__name__}: {x.shape}")
        return x