"""Provides class definition and methods for AffineRegistration."""

import tensorflow as tf

from ..pretrain import pretrain_model
from ..stn.affine.st_affine import SpatialTransformerAffine
from ..train import train_model
from .LocNet import LocNet
from .TransformationExtractor import TransformationExtractor

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
            pretrain_strategy = "identity",
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
        self.pretrain_strategy = pretrain_strategy

        self.regularize = regularize
        self.reg_weight = reg_weight

        self.locnet = LocNet(
            input_shape = self.img_res,
            initial_filters = 32,
            min_output_size = 10
        )
        self.transformation_extractor = TransformationExtractor(
            units = 5,  # 5 params: scale_x, scale_y, rotation, trans_x, trans_y
            input_shape = self.img_res,
            locnet = self.locnet,
            factor = factor
        )
        self.transformer = SpatialTransformerAffine(
            img_res = (self.img_res[1], self.img_res[2]),
            out_dims = (self.img_res[1], self.img_res[2]),
            B = self.B)

    def call(self):
        xs = self.moving
        xs = self.locnet(inputs=xs)
        xs = tf.transpose(xs, [0, 3, 1, 2])
        theta = self.transformation_extractor(inputs=xs)

        # Store raw parameters for regularization
        self.theta_raw = theta

        # Extract parameters with gentler initialization
        # Don't use tanh initially - it can cause gradient vanishing
        scale_x = 1.0 + theta[:, 0] * 0.1    # Start with small deviations
        scale_y = 1.0 + theta[:, 1] * 0.1
        rotation = theta[:, 2] * 0.1          # About ±5.7 degrees initially
        trans_x = theta[:, 3] * 0.05          # Small translations
        trans_y = theta[:, 4] * 0.05

        # Clip to reasonable ranges to prevent extreme values
        scale_x = tf.clip_by_value(scale_x, 0.5, 2.0)
        scale_y = tf.clip_by_value(scale_y, 0.5, 2.0)
        rotation = tf.clip_by_value(rotation, -0.785, 0.785)  # ±45 degrees
        trans_x = tf.clip_by_value(trans_x, -0.5, 0.5)
        trans_y = tf.clip_by_value(trans_y, -0.5, 0.5)

        # Store individual parameters for inspection/debugging
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.rotation = rotation
        self.trans_x = trans_x
        self.trans_y = trans_y

        # Construct affine matrix without shear
        cos_r = tf.cos(rotation)
        sin_r = tf.sin(rotation)

        # Add small epsilon to prevent numerical issues
        eps = 1e-7

        a11 = scale_x * cos_r + eps
        a12 = -scale_y * sin_r
        a13 = trans_x
        a21 = scale_x * sin_r
        a22 = scale_y * cos_r + eps
        a23 = trans_y

        theta1 = tf.stack([a11, a12, a13], axis=1)
        theta2 = tf.stack([a21, a22, a23], axis=1)
        self.theta = tf.stack([theta1, theta2], axis=1)
        self.theta = tf.squeeze(self.theta, axis=0)

        self.moving_hat = self.transformer(
            input_fmap=self.moving,
            theta=self.theta,
            B=self.B)

    def __call__(self):
        self.call()

    def train(
            self,
            loss_type: str = "mi",
            optim: tf.keras.optimizers = None,
            ITERMAX: int = 1000,
            patience: int = 100,
            lr_schedule_type: str = "cosine_restarts"
    ):
        """
        Train the AffineRegistration model.
        
        Args:
            loss_type: Loss function type ("mi" or "mse")
            optim: Keras optimizer. If None, creates Adam with specified schedule
            ITERMAX: Maximum iterations
            patience: Early stopping patience
            lr_schedule_type: LR schedule when optim=None ("cosine_restarts", "cosine", "exponential", "constant")
        """
        # Create default optimizer if not provided
        if optim is None:
            if lr_schedule_type == "cosine_restarts":
                lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
                    initial_learning_rate=0.01,
                    first_decay_steps=100,
                    t_mul=1.5,
                    m_mul=0.8,
                    alpha=0.0001
                )
            elif lr_schedule_type == "cosine":
                lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
                    initial_learning_rate=0.01,
                    decay_steps=1000,  # Fixed, not ITERMAX
                    alpha=0.001
                )
            elif lr_schedule_type == "exponential":
                lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=0.01,
                    decay_steps=100,
                    decay_rate=0.96
                )
            else:  # "constant"
                lr_schedule = 0.01

            optim = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

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
                learning_rate = self.pretrain_lr,
                strategy = self.pretrain_strategy
            )

        self.loss_list = list()
        train_model(
            self,
            loss_type=loss_type,
            optim=optim,
            ITERMAX=ITERMAX,
            patience=patience
        )

