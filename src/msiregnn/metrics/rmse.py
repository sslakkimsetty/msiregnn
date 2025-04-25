"""Evaluates RMSE between two images that is differentiable."""

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
    # se = tf.square(x.numpy().squeeze() - y.numpy().squeeze())
    se = tf.square(x - y)
    mse = tf.reduce_mean(se)
    return tf.sqrt(mse)
