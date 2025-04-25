"""Provides class definition and methods for BsplineRegistration."""

import numpy as np
import tensorflow as tf
from .LocNet import LocNet
from .TransformationExtractor import TransformationExtractor
from ..stn.bspline.st_bspline import SpatialTransformerBspline
from ..train import train_model
from ..pretrain import pretrain_model

__all__ = [
    "BsplineRegistration"
]


class BsplineRegistration(tf.keras.models.Model):
    """Class definition for BsplineRegistration model.
        :param fixed: reference image.
        :param moving: target image to be transformed.
    """

    def __init__(
            self,
            fixed: tf.Tensor = tf.ones(shape=(1, 200, 200, 1)),
            moving: tf.Tensor = tf.ones(shape=(1, 200, 200, 1)),
            factor = 1,
            pretrain = False,
            theta_id = None,
            pretrain_epochs = 300,
            pretrain_lr = 0.001,
            regularize=True,
            reg_weight=1e-3
    ):
        super(BsplineRegistration, self).__init__()
        self.fixed = fixed
        self.moving = moving
        self.B = 1
        self.img_res = self.moving.shape
        self.grid_res = (23, 21)

        self.pretrain = pretrain
        self.theta_id = theta_id
        self.pretrain_epochs = pretrain_epochs
        self.pretrain_lr = pretrain_lr

        self.regularize = regularize
        self.reg_weight = reg_weight

        self.locnet = LocNet(
            input_shape = self.img_res,
            initial_filters = 32,
            min_output_size = 10
        )

        self.grid_res = (self.grid_res[0] * np.sqrt(factor).astype(np.int32),
                         self.grid_res[1] * np.sqrt(factor).astype(np.int32)) # accounting for factor
        self.transformation_extractor = TransformationExtractor(
            units = 2 * self.grid_res[0] * self.grid_res[1],
            input_shape = self.img_res,
            locnet = self.locnet,
            factor = factor)

        self.transformer = SpatialTransformerBspline(
            img_res = (self.img_res[1], self.img_res[2]),
            grid_res = self.grid_res,
            out_dims = (self.img_res[1], self.img_res[2]),
            B = self.B)

    def call(self):
        xs = self.moving
        xs = self.locnet(inputs=xs)
        xs = tf.transpose(xs, [0, 3, 1, 2])
        theta = self.transformation_extractor(inputs=xs)
        self.theta = tf.reshape(theta, (self.B, 2, self.grid_res[0], self.grid_res[1]))
        self.moving_hat, self.delta_hat = self.transformer(
            input_fmap=self.moving,
            theta=self.theta,
            B=self.B)

    def __call__(self):
        self.call()

    def train(
            self,
            loss_type: str = "mi",
            optim: tf.keras.optimizers = tf.keras.optimizers.Adagrad(learning_rate=1e-3),
            ITERMAX: int = 1000  # noqa
    ):
        if self.pretrain:
            if not self.theta_id:
                theta_id_shape = (self.B, 2, self.grid_res[0], self.grid_res[1])
                self.theta_id = tf.zeros(
                    shape = theta_id_shape,
                    dtype=tf.float32
                )
            pretrain_model(
                self,
                theta_id = self.theta_id,
                epochs = self.pretrain_epochs,
                learning_rate = self.pretrain_lr)

        self.loss_list = list()
        train_model(self, loss_type=loss_type, optim=optim, ITERMAX=ITERMAX)

