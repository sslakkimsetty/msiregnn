"""Provides automated LocNet layers for *Registration model."""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, LeakyReLU, MaxPooling2D

__all__ = [
    "LocNet"
]

class LocNet(tf.keras.layers.Layer):
    """
    LocNet: A custom tensorflow layer for capturing spatial patterns in image registration tasks.

    This class defines a convolutional neural network (CNN) architecture with varying kernel sizes,
    activation functions, and pooling layers to effectively capture spatial patterns in images.
    The network is designed to be used in image registration tasks, where it helps in aligning
    images by learning spatial transformations.

    Attributes:
        input_shape (tuple): The size of the input image (width, height).
        initial_filters (int): The number of filters in the first convolutional layer.
        layers_list (list): A list to store the layers of the network.

    Methods:
        __init__(self, input_shape, initial_filters=32): Initializes the LocNet with the given input size and initial filters.
        call(self, inputs): Defines the forward pass of the network.

    Output:
    tf.Tensor: The transformed image tensor after passing through the network layers.
    """

    def __init__(
            self,
            input_shape = (1, 200, 200, 1),
            initial_filters=32,
            min_output_size=10):

        super(LocNet, self).__init__()
        self.layers_list = []
        self.current_width, self.current_height = input_shape[1], input_shape[2]
        num_filters = initial_filters
        self.layer_count = 0
        self.min_output_size = min_output_size
        self.activation = "relu"

        self.PARAMS = {
            "initial": {
                "kernel_size": 3,
                "kernel_stride": 1,
                "pool_size": 2,
                "pool_stride": 1,
                "activation": "relu"
            },
            "intermediate": {
                "kernel_size": 5,
                "kernel_stride": 2,
                "pool_size": 3,
                "pool_stride": 2,
                "activation": LeakyReLU(negative_slope=0.1)
            },
            "final": {
                "kernel_size": 7,
                "kernel_stride": 2,
                "pool_size": 4,
                "pool_stride": 3,
                "activation": LeakyReLU(negative_slope=0.1)
            }
        }
        self.params = self.PARAMS["initial"]

        while self.is_next_layer_feasible(): # Check if adding another layer is feasible
            self.layers_list.append(
                Conv2D(
                    filters = num_filters,
                    kernel_size = self.params["kernel_size"],
                    activation = self.params["activation"],
                    strides = self.params["kernel_stride"]))

            self.current_width = (self.current_width - self.params["kernel_size"]) // self.params["kernel_stride"] + 1
            self.current_height = (self.current_height - self.params["kernel_size"]) // self.params["kernel_stride"] + 1

            self.layers_list.append(
                MaxPooling2D(
                    pool_size = self.params["pool_size"],
                    strides = self.params["pool_stride"]))

            self.current_width = (self.current_width - self.params["pool_size"]) // self.params["pool_stride"] + 1
            self.current_height = (self.current_height - self.params["pool_size"]) // self.params["pool_stride"] + 1

            num_filters *= 2
            self.layer_count += 1

        self.layers_list.append(Conv2D(filters=2, kernel_size=1, activation="sigmoid", strides=1))

    def is_next_layer_feasible(self):
        if self.layer_count == 3:
            self.params = self.PARAMS["intermediate"]
        elif self.layer_count == 6:
            self.params = self.PARAMS["final"]

        return (self._is_next_layer_feasible(self.current_width) and
                self._is_next_layer_feasible(self.current_height))

    def _is_next_layer_feasible(self, current_size=100):
        before_pool_offset = self.min_output_size - 1
        before_pool_stride = before_pool_offset * self.params["pool_stride"]
        before_pool_size = before_pool_stride + self.params["pool_size"]

        before_kernel_offset = before_pool_size - 1
        before_kernel_stride = before_kernel_offset * self.params["kernel_stride"]
        before_kernel_size = before_kernel_stride + self.params["kernel_size"]
        return before_kernel_size <= current_size

    def call(self, inputs):
        xs = inputs
        for layer in self.layers_list:
            xs = layer(xs)
        return xs

    def __call__(self, inputs):
        return self.call(inputs)










