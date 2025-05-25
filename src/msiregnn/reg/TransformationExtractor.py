"""Provides automated TransformationExtractor layers for *Registration model."""

import functools

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten

from ..utils import locnet_output_shape
from .LocNet import LocNet

__all__ = [
    "TransformationExtractor"
]


class TransformationExtractor(tf.keras.layers.Layer):
    """
    TransformationExtractor: A custom TensorFlow layer for extracting transformation parameters.

    This class defines a neural network layer block that processes the output of a convolutional neural network (CNN) to extract transformation parameters. It consists of two fully connected (dense) layers that process the flattened input tensor.

    Attributes:
        units (int): The number of units in the first dense layer.
        factor (int): A factor to multiply the number of units in the second dense layer.

    Methods:
        __init__(self, units=100, factor=1): Initializes the TransformationExtractor with the given number of units and factor.
        call(self, inputs): Defines the forward pass of the layer.

        # Example usage
        model = TransformationExtractor(units=100, factor=1)
        input_tensor = tf.random.uniform((1, 20, 20, 2))  # Example input tensor
        output = model(input_tensor)
        print(output)
        ```
    """

    def __init__(
            self,
            units = 100,
            input_shape = (1, 200, 200, 1),
            locnet = LocNet(),
            factor = 1
    ):
        super(TransformationExtractor, self).__init__()

        self.layers_list = []
        shape = locnet_output_shape(locnet, input_shape=input_shape)
        locnet_size = functools.reduce(lambda x, y: x*y, shape)
        locnet_layers_num = len(locnet.layers_list) // 2 # Conv2D + MaxPooling2D
        layers_units = self.units_calc(locnet_size, locnet_layers_num, units, factor)

        self.layers_list.append(Flatten())
        [self.layers_list.append(Dense(units=x)) for x in layers_units[::-1]]

        def parameter_aware_initializer(shape, dtype=tf.float32):
            """Initialize with different scales for different parameters."""
            # Base initialization
            weights = tf.random.normal(shape, mean=0.0, stddev=0.1, dtype=dtype)

            # Only apply parameter-specific scaling if output is 5 (affine params)
            if shape[1] == 5:
                # [scale_x, scale_y, rotation, trans_x, trans_y]
                param_scales = tf.constant([
                    [1.0],
                    [1.0],
                    [2.0],    # rotation - more variance for better exploration
                    [0.5],    # trans_x - less variance
                    [0.5]     # trans_y - less variance
                ], dtype=dtype)

                # Broadcast across all input features
                param_scales = tf.transpose(param_scales)
                weights = weights * param_scales
            return weights

        self.layers_list.append(
            Dense(
                units=layers_units[-1]*factor,
                kernel_initializer=parameter_aware_initializer,
                bias_initializer="zeros"
            )
        )

    def units_calc(self, locnet_size, locnet_layers_num, out_units, factor):
        N = locnet_size / out_units

        if locnet_layers_num <= 3:
            out = [np.sqrt(out_units*N), out_units]
        elif locnet_layers_num <= 6:
            out = [np.cbrt(N), np.square(np.cbrt(N)), out_units]
        else:
            out = [N**(1/4), N**(1/2), N**(3/4), out_units]

        return [round(x) for x in out]

    def call(self, inputs):
        xs = inputs
        for layer in self.layers_list:
            xs = layer(xs)
        return xs

    def __call__(self, inputs):
        return self.call(inputs)





