"""Evaluates RMSE between two images that is differentiable."""

import numpy as np
import tensorflow as tf

__all__ = [
    "rmse"
]


def rmse(
        x: tf.Tensor,
        y: tf.Tensor
) -> float:
    """
    Compute the root-mean-squared error between two equal sized Tensors.

    :param x: A Tensor
    :param y: Equal sized Tensor to :param x:

    :return: the root mean squared error between the Tensors
    """
    se = np.square(x.numpy().squeeze() - y.numpy().squeeze())
    mse = se.mean()
    return np.sqrt(mse)
