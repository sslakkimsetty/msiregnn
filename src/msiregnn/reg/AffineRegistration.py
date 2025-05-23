"""Provides class definition and methods for AffineRegistration."""

import msiregnn as msn
import numpy as np
import tensorflow as tf 

from .LocNet import LocNet
from .TransformationExtractor import TransformationExtractor
from ..stn.affine.st_affine import SpatialTransformerAffine
from ..train import train_model
from ..pretrain import pretrain_model

__all__ = [
    "AffineRegistration"
]


class AffineRegistration(tf.keras.models.Model):
    """Class definition for AffineRegistration model.
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
        super(AffineRegistration, self).__init__()
        self.fixed = fixed
        self.moving = moving
        self.B = 1
        self.img_res = self.moving.shape

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
        self.transformation_extractor = TransformationExtractor(
            units = 4,
            input_shape = self.img_res,
            locnet = self.locnet,
            factor = factor)
        self.transformer = SpatialTransformerAffine(
            img_res = (self.img_res[1], self.img_res[2]),
            out_dims = (self.img_res[1], self.img_res[2]),
            B = self.B)

    def call(self):
        xs = self.moving
        xs = self.locnet(inputs=xs)
        xs = tf.transpose(xs, [0, 3, 1, 2])
        theta = self.transformation_extractor(inputs=xs)
        self.theta_sterile = theta

        a = theta[:, 0]
        b = theta[:, 1]
        c = 0.5 + theta[:, 2]
        d = 0.5 + theta[:, 3]

        _a = a * tf.math.cos(b)
        _b = -a * tf.math.sin(b)
        _c = ( (1 - a * tf.math.cos(b)) * c ) + ( a * tf.math.sin(b) * d )

        _d = -_b
        _e = _a
        _f = ( (1 - a * tf.math.cos(b)) * d ) - ( a * tf.math.sin(b) * c )

        theta1 = tf.stack([_a, _b, _c], axis=1)
        theta2 = tf.stack([_d, _e, _f], axis=1)
        theta = tf.stack([theta1, theta2], axis=1)
        self.theta = tf.squeeze(theta, axis=0)

        self.moving_hat = self.transformer(
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
                self.theta_id = tf.constant(
                    [
                        [1, 0, 0],
                        [0, 1, 0]], dtype=tf.float32
                )
            pretrain_model(
                self,
                theta_id = self.theta_id,
                epochs = self.pretrain_epochs,
                learning_rate = self.pretrain_lr)

        self.loss_list = list()
        train_model(self, loss_type=loss_type, optim=optim, ITERMAX=ITERMAX)

