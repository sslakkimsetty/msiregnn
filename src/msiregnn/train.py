"""Trains a coregistration model and evaluates transformation parameters."""

import tensorflow as tf
from .utils import imshow
from .metrics import mi, rmse

__all__ = [
    "train_model"
]


def train_model(
        model,
        loss_type: str = "mi",
        optim: tf.keras.optimizers = tf.keras.optimizers.Adagrad(learning_rate=1e-3),
        ITERMAX: int = 1000, # noqa
):
    """
    Train the AffineRegistration model.
    :param loss_type: Loss type as str.
    :param optim: optimizer used for training.
    :param ITERMAX: maximum number of iterations.

    :return: a tuple consisting of trained BsplineRegistration model and a loss list.
    """
    it = 0 # iteration count
    model.loss_list = list() # list to store loss values

    while True:
        with tf.GradientTape() as tape:
            loss_value, grads = grad(model, loss_type=loss_type)
            model.loss_list.append(loss_value.numpy())
            optim.apply_gradients(zip(grads, model.trainable_variables))

        it += 1
        if it % 25 == 0:
            print("ITER:", it, "LOSS:", loss_value.numpy())
            # print("theta:", model.theta.numpy())
            imshow(model.moving_hat)
        if it >= ITERMAX:
            break


def grad(
        model,
        loss_type: str = "mi",
) -> tf.Tensor:
    """
    Compute gradients for backprop in AffineRegistration

    :param loss_type: Loss type as str

    :return: Computed gradients of the loss to be backproped as tf.Tensors.
    """
    # Initialize maximum regularization loss to model loss
    max_reg_ratio = 0.1

    with tf.GradientTape() as tape:
        model.call()
        model_loss = _loss(model, loss_type=loss_type)

        if model.regularize:
            reg_loss = regularization_loss(model)
            reg_ratio = (model.reg_weight*reg_loss) / tf.abs(model_loss)

            # Adjust the regularization weight based on the ratio
            if reg_ratio > max_reg_ratio:
                model.reg_weight *= max_reg_ratio / reg_ratio

            total_loss = model_loss + model.reg_weight * reg_loss
    return total_loss, tape.gradient(total_loss, model.trainable_variables)


def _loss(
        model,
        loss_type: str = "mi"
) -> tf.Tensor:
    """
    Compute the loss metric between two images as tf.Tensors

    :param loss_type: Loss type as str

    :return: The loss value as tf.Tensor.
    """
    if loss_type == "mi":
        loss_object = mi
        return -loss_object(model.fixed, model.moving_hat, n=500)
    elif loss_type == "mse":
        loss_object = rmse
        return loss_object(model.fixed, model.moving_hat)


def regularization_loss(model): # noqa
    """
    Compute the regularization loss for the model.

    :param model: The model instance.

    :return: The regularization loss value as tf.Tensor.
    """
    if model.__class__.__name__ == "AffineRegistration":
        return affine_model_regularization_loss(model)
    elif model.__class__.__name__ == "BsplineRegistration":
        return bspline_model_regularization_loss(model)


def affine_model_regularization_loss(model):
    target = tf.constant([1.0, 0.0, 0.0, 0.0], dtype=tf.float32)
    diff = model.theta_sterile - target
    reg_loss = (tf.pow(tf.abs(diff[0]) + 1, 3) +
                tf.reduce_sum(tf.square(diff[1:] + 1)))
    return tf.reduce_mean(tf.square(model.theta_sterile - target))


def bspline_model_regularization_loss(model):
    # Clip the tensor elements to 1
    clipped_theta = tf.clip_by_value(
        model.theta,
        clip_value_min = -1,
        clip_value_max = 1
    )

    # Compute the regularization loss
    loss = tf.reduce_sum(tf.pow(tf.math.abs(model.theta), 3))

    # Subtract the sum of the original clipped theta from the loss
    # This is to ensure that the loss is not too large
    loss -= tf.reduce_sum(clipped_theta)
    return loss