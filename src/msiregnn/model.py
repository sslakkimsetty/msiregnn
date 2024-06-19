"""The coregistration model and training loop."""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Flatten,
    MaxPooling2D,
)

import msiregnn as msn

__all__ = [
    "LocNet",
    "TransformationRegressor",
    "BsplineRegistration"
]


# class LocNet(tf.keras.layers.Layer):
#
#     def __init__(self):
#         super(LocNet, self).__init__()
#
#         self.conv1 = Conv2D(filters=32, kernel_size=5, activation="relu", strides=2)
#         self.avgp1 = MaxPooling2D(pool_size=(2,2), strides=(1,1))
#         self.conv2 = Conv2D(filters=32, kernel_size=5, activation="relu", strides=2)
#         self.avgp2 = MaxPooling2D(pool_size=(2,2), strides=(1,1))
#         self.conv3 = Conv2D(filters=32, kernel_size=5, activation="relu", strides=1)
#         self.avgp3 = MaxPooling2D(pool_size=(2,2), strides=(1,1))
#         self.conv4 = Conv2D(filters=32, kernel_size=5, activation="relu", strides=1)
#         self.avgp4 = MaxPooling2D(pool_size=(2,2), strides=(1,1))
#         self.conv5 = Conv2D(filters=32, kernel_size=5, activation="relu", strides=2)
#         self.conv6 = Conv2D(filters=32, kernel_size=5, activation="relu", strides=1)
#         self.conv7 = Conv2D(filters=32, kernel_size=1, activation=None, strides=1)
#         self.conv8 = Conv2D(filters=2, kernel_size=1, activation="sigmoid", strides=1)
#
#     def call(self, inputs):
#         x = self.conv1(inputs)
#         x = self.avgp1(x)
#         x = self.conv2(x)
#         x = self.avgp2(x)
#         # x = self.conv3(x)
#         # x = self.avgp3(x)
#         # x = self.conv4(x)
#         # x = self.avgp4(x)
#         x = self.conv5(x)
#         x = self.conv6(x)
#         x = self.conv7(x)
#         x = self.conv8(x)
#         return x


class LocNet(tf.keras.layers.Layer):

    def __init__(self):
        super(LocNet, self).__init__()

        self.conv1 = Conv2D(filters=32, kernel_size=15, activation="relu", strides=3)
        self.avgp1 = MaxPooling2D(pool_size=(8,8), strides=(1,1))
        self.conv2 = Conv2D(filters=32, kernel_size=15, activation="relu", strides=3)
        self.avgp2 = MaxPooling2D(pool_size=(8,8), strides=(1,1))
        self.conv3 = Conv2D(filters=32, kernel_size=15, activation="relu", strides=3)
        self.avgp3 = MaxPooling2D(pool_size=(2,2), strides=(1,1))
        self.conv4 = Conv2D(filters=32, kernel_size=15, activation="relu", strides=1)
        self.avgp4 = MaxPooling2D(pool_size=(2,2), strides=(1,1))
        self.conv5 = Conv2D(filters=32, kernel_size=15, activation="relu", strides=3)
        self.conv6 = Conv2D(filters=32, kernel_size=15, activation="relu", strides=2)
        self.conv7 = Conv2D(filters=32, kernel_size=1, activation=None, strides=1)
        self.conv8 = Conv2D(filters=2, kernel_size=1, activation="sigmoid", strides=1)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.avgp1(x)
        x = self.conv2(x)
        x = self.avgp2(x)
        # x = self.conv3(x)
        # x = self.avgp3(x)
        # x = self.conv4(x)
        # x = self.avgp4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        return x


class TransformationRegressor(tf.keras.layers.Layer):

    def __init__(self, units=100, factor=1):
        super(TransformationRegressor, self).__init__()

        self.flatten = Flatten()
        self.fc1 = Dense(units=units)
        self.fc2 = Dense(units=units*factor, bias_initializer="zeros")

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class BsplineRegistration(tf.keras.models.Model):

    def __init__(self, img_res, factor=1):
        super(BsplineRegistration, self).__init__()
        self.B = 1
        self.img_res = img_res
        self.loc_net = LocNet()

        _test_arr = np.ones((1,img_res[0],img_res[1],1)).astype(np.float32)
        _test_arr = self.loc_net(_test_arr)
        self.grid_res = _test_arr.numpy().squeeze().shape
        self.grid_res = [self.grid_res[0], self.grid_res[1]]
        print("Grid res:", self.grid_res)

        self.transformer_regressor = TransformationRegressor(units=self.grid_res[0]*self.grid_res[1]*2, factor=factor)
        self.grid_res = [self.grid_res[0]*np.sqrt(factor).astype(np.int32), self.grid_res[1]*np.sqrt(factor).astype(np.int32)]
        self.transformer = msn.SpatialTransformerBspline(img_res=img_res,
                                                         grid_res = self.grid_res,
                                                         out_dims=img_res,
                                                         B=self.B)

    def call(self, moving):
        xs = moving
        xs = self.loc_net(xs)
        xs = tf.transpose(xs, [0, 3, 1, 2])
        # theta = xs
        theta = self.transformer_regressor(xs)
        theta = tf.reshape(theta, (self.B, 2, self.grid_res[0], self.grid_res[1]))
        xt = self.transformer(moving, theta, self.B)
        return xt, theta
