""" Methods for coregistration loop. """

import numpy as np
import tensorflow as tf
import msiregnn as msn

__all__ = [
    "loss",
    "grad",
    "train"
]


def loss(
        x: tf.Tensor = tf.ones(shape=(1, 200, 200, 1)),
        y: tf.Tensor = tf.ones(shape=(1, 200, 200, 1)),
        loss_type: str = "mi"
) -> tf.Tensor:
    """
    Compute the loss metric between two images as tf.Tensors

    :param x: Image 1.
    :param y: Image 2.
    :param loss_type: Loss type as str

    :return: The loss value as tf.Tensor.
    """
    if loss_type == "mi":
        loss_object = msn.metrics.mi
        return -loss_object(x, y, n=500)
    elif loss_type == "mse":
        loss_object = msn.metrics.rmse
        return loss_object(x, y)


def grad(
        model: msn.BsplineRegistration,
        fixed: tf.Tensor = tf.ones(shape=(1,200,200,1)),
        moving: tf.Tensor = tf.ones(shape=(1,200,200,1)),
        loss_type: str = "mi"
) -> tf.Tensor:
    """
    Compute gradients for backprop in BsplineRegistration

    :param model: the BsplineRegistration model.
    :param fixed: reference image (must be the same one used in the model training).
    :param moving: target image (must be the same one used in the model training).
    :param loss_type: Loss type as str

    :return: Computed gradients of the loss to be backproped as tf.Tensors.
    """
    with tf.GradientTape() as tape:
        (_moving, __), _ = model(moving)
        loss_value = loss(fixed, _moving, loss_type=loss_type)
    return loss_value, tape.gradient(loss_value, model.trainable_variables), _


def train(
        model: msn.BsplineRegistration,
        fixed: tf.Tensor = tf.ones(shape=(1, 200, 200, 1)),
        moving: tf.Tensor = tf.ones(shape=(1, 200, 200, 1)),
        loss_type: str = "mi",
        optim: tf.keras.optimizers = tf.keras.optimizers.Adagrad(learning_rate=1e-3),
        ITERMAX: int = 1000 # noqa
) -> tuple[msn.BsplineRegistration, list]:
    """
    Train the BsplineRegistration model.

    :param model: the BsplineRegistration model.
    :param fixed: reference image.
    :param moving: target image to be transformed.
    :param loss_type: Loss type as str.
    :param optim: optimizer used for training.
    :param ITERMAX: maximum number of iterations.

    :return: a tuple consisting of trained BsplineRegistration model and a loss list.
    """
    it = 0
    loss_list = list()

    while True:
        with tf.GradientTape() as tape:
            loss_value, grads, _ = grad(model, fixed, moving, loss_type=loss_type)
            loss_list.append(loss_value.numpy())
            optim.apply_gradients(zip(grads, model.trainable_variables))

        it += 1
        if it % 100 == 0:
            print("ITER:", it)
        if it >= ITERMAX:
            break
    return model, loss_list

