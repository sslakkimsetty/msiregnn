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
        lr_schedule: bool = True,
        patience: int = 50,  # For early stopping
        lr_decay_factor: float = 0.1  # Controls how much LR decreases over full training
):
    """
    Train the AffineRegistration model with adaptive learning rate scheduling.
    
    :param model: The model to train
    :param loss_type: Loss type as str
    :param optim: Optimizer used for training
    :param ITERMAX: Maximum number of iterations
    :param lr_schedule: Whether to use learning rate scheduling
    :param patience: Number of iterations to wait for improvement before early stopping
    :param lr_decay_factor: Controls how much learning rate decreases by end of training (e.g., 0.1 means final LR will be ~10% of initial LR)
    
    :return: Trained model
    """
    it = 0
    model.loss_list = list()
    best_loss = float('inf')
    patience_counter = 0
    initial_lr = optim.learning_rate.numpy()
    
    # Calculate adaptive decay rate based on ITERMAX
    adaptive_decay = lr_decay_factor / ITERMAX
    
    # How often to update LR (e.g., 10 times during training)
    update_frequency = max(1, ITERMAX // 10)
    
    while True:
        with tf.GradientTape() as tape:
            loss_value, grads = grad(model, loss_type=loss_type) 

            if tf.math.is_nan(loss_value):
                print(f"NaN detected at iteration {it}")
                print("Parameters at NaN:")
                if hasattr(model, 'debug_parameters'):
                    model.debug_parameters()
                break

            model.loss_list.append(loss_value.numpy())
            
            # Learning rate scheduling with adaptive parameters
            if lr_schedule and it > 0 and it % update_frequency == 0:
                # Decay factor now scales with ITERMAX
                current_lr = initial_lr * (1.0 / (1.0 + adaptive_decay * it))
                optim.learning_rate.assign(current_lr)
            
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
            print(f"ITER: {it}, LOSS: {loss_value.numpy()}, LR: {optim.learning_rate.numpy()}")
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