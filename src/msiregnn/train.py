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
        ITERMAX: int = 1000,
        patience: int = 100,
        lr_reduction_patience: int = 50,  # For plateau detection
        lr_reduction_factor: float = 0.5   # For plateau reduction
):
    """
    Train the AffineRegistration model.
    
    Args:
        model: Model to train
        loss_type: Loss function type ("mi" or "mse")
        optim: Keras optimizer (should include LR schedule if desired)
        ITERMAX: Maximum iterations
        patience: Early stopping patience
        lr_reduction_patience: Iterations before reducing LR on plateau
        lr_reduction_factor: Factor to reduce LR by on plateau
    """
    it = 0
    model.loss_list = list()
    best_loss = float('inf')
    patience_counter = 0
    plateau_counter = 0
    plateau_best_loss = float('inf')
    
    while True:
        with tf.GradientTape() as tape:
            loss_value, grads = grad(model, loss_type=loss_type)
            model.loss_list.append(loss_value.numpy())
            
            # Plateau detection for fixed LR optimizers
            if loss_value < plateau_best_loss:
                plateau_best_loss = loss_value
                plateau_counter = 0
            else:
                plateau_counter += 1
                
            # Reduce LR on plateau (only works for fixed LR)
            if (plateau_counter >= lr_reduction_patience and 
                isinstance(optim.learning_rate, (int, float))):
                current_lr = float(optim.learning_rate)
                new_lr = current_lr * lr_reduction_factor
                optim.learning_rate.assign(new_lr)
                plateau_counter = 0
                print(f"Plateau detected. Reducing LR from {current_lr:.6f} to {new_lr:.6f}")
            
            optim.apply_gradients(zip(grads, model.trainable_variables))
            
            # Early stopping
            if loss_value < best_loss:
                best_loss = loss_value
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at iteration {it}")
                break

        it += 1
        if it % 25 == 0:
            current_lr = (optim.learning_rate.numpy() if hasattr(optim.learning_rate, 'numpy') 
                         else optim.learning_rate)
            print(f"ITER: {it}, LOSS: {loss_value.numpy()}, LR: {current_lr}")
            imshow(model.moving_hat)
        if it >= ITERMAX:
            break
            
    return model


def grad(
        model,
        loss_type: str = "mi",
        clip_gradients: bool = True,
        max_grad_norm: float = 1.0,
) -> tf.Tensor:
    """
    Compute gradients for backprop in AffineRegistration

    :param model: The model to compute gradients for
    :param loss_type: Loss type as str
    :param clip_gradients: Whether to clip gradients to prevent exploding gradients
    :param max_grad_norm: Maximum norm for gradient clipping when clip_gradients is True

    :return: A tuple of (total_loss, gradients) where gradients are computed (and optionally clipped)
             tf.Tensors to be used with an optimizer.
    """
    # Initialize maximum regularization loss to model loss
    max_reg_ratio = 0.5

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
        else:
            total_loss = model_loss

    # Compute gradients
    gradients = tape.gradient(total_loss, model.trainable_variables)
    
    # Apply gradient clipping if enabled
    if clip_gradients:
        gradients, _ = tf.clip_by_global_norm(gradients, max_grad_norm)
        
    return total_loss, gradients


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
    if hasattr(model, 'theta_raw'):
        # For new 5-parameter model: penalize deviation from zeros
        # When theta_raw is all zeros, we get identity transformation
        return tf.reduce_mean(tf.square(model.theta_raw))
    else:
        # Fallback for old 4-parameter model
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