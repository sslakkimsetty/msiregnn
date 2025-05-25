"""Trains a coregistration model and evaluates transformation parameters."""

import tensorflow as tf
from .utils import imshow
from .metrics import mi, rmse

__all__ = [
    "train_model"
]


def train_model(
        model: tf.keras.models.Model,
        loss_type: str = "mi",
        optim: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adagrad(learning_rate=1e-3),
        ITERMAX: int = 1000,
        patience: int = 100,
        lr_reduction_patience: int = 50,  # For plateau detection
        lr_reduction_factor: float = 0.5   # For plateau reduction
) -> tf.keras.models.Model:
    """
    Train a registration model using gradient descent with early stopping and learning rate reduction.
    
    This function implements a training loop with early stopping based on loss improvement
    and optional learning rate reduction on plateau. It supports both mutual information (MI)
    and mean squared error (MSE) loss functions.
    
    Args:
        model: The registration model to train. Must have attributes:
            - fixed: The fixed/reference image tensor
            - moving_hat: The transformed moving image (computed by model.call())
            - loss_list: List to store loss values during training
            - trainable_variables: Model parameters to optimize
        loss_type: Type of loss function to use. Options:
            - "mi": Mutual Information (negative MI is minimized)
            - "mse": Mean Squared Error (RMSE)
        optim: TensorFlow/Keras optimizer instance. Should include learning rate
            schedule if adaptive learning rate is desired. Default is Adagrad
            with learning rate of 1e-3.
        ITERMAX: Maximum number of training iterations. Training stops when this
            limit is reached even if convergence criteria are not met.
        patience: Number of iterations to wait for loss improvement before early
            stopping. If loss doesn't improve for this many iterations, training stops.
        lr_reduction_patience: Number of iterations without improvement before
            reducing learning rate. Only applies to optimizers with fixed learning
            rates (not schedules).
        lr_reduction_factor: Factor by which to reduce learning rate on plateau.
            New LR = old LR * lr_reduction_factor. Only applies to fixed LR optimizers.
    
    Returns:
        The trained model with updated parameters and populated loss_list attribute.
    
    Side Effects:
        - Updates model parameters in-place
        - Populates model.loss_list with loss values from each iteration
        - Prints progress every 25 iterations showing iteration number, loss, and current LR
        - Displays the transformed image (model.moving_hat) every 25 iterations
        - Prints early stopping message if patience is exceeded
        - Prints learning rate reduction messages when plateau is detected
    
    Note:
        The function uses gradient clipping and regularization if enabled in the model.
        Plateau detection only works with optimizers that have a fixed learning rate
        (not those using learning rate schedules).
    
    Example:
        >>> model = AffineRegistration(fixed=fixed_img, moving=moving_img)
        >>> optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        >>> trained_model = train_model(
        ...     model, 
        ...     loss_type="mi",
        ...     optim=optimizer,
        ...     ITERMAX=500,
        ...     patience=50
        ... )
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
        model: tf.keras.models.Model,
        loss_type: str = "mi",
        clip_gradients: bool = True,
        max_grad_norm: float = 1.0,
) -> tuple[tf.Tensor, list[tf.Tensor]]:
    """Compute loss and gradients for backpropagation in registration models.
    
    This function computes the total loss (including optional regularization) and its
    gradients with respect to the model's trainable variables. It supports adaptive
    regularization weight adjustment and gradient clipping to prevent training instability.
    
    Args:
        model: The registration model to compute gradients for. Must have:
            - call(): Method to perform forward pass
            - trainable_variables: List of variables to compute gradients for
            - regularize: Boolean flag indicating if regularization is enabled
            - reg_weight: Float weight for regularization term (if regularize=True)
        loss_type: Type of loss function to use. Options:
            - "mi": Mutual Information (negative MI is minimized)
            - "mse": Mean Squared Error (RMSE)
        clip_gradients: Whether to apply gradient clipping to prevent exploding
            gradients. Recommended for stable training.
        max_grad_norm: Maximum global norm for gradient clipping when clip_gradients
            is True. Gradients are scaled down if their global norm exceeds this value.
    
    Returns:
        A tuple containing:
            - total_loss: The computed loss value (model loss + regularization) as tf.Tensor
            - gradients: List of gradient tensors corresponding to model.trainable_variables,
              optionally clipped if clip_gradients=True
    
    Notes:
        - The function automatically adjusts regularization weight if it becomes too
          dominant (more than 50% of the model loss)
        - Gradient clipping uses global norm clipping across all gradients
        - The model's call() method is executed within the gradient tape context
    
    Example:
        >>> model = AffineRegistration(fixed=fixed_img, moving=moving_img)
        >>> loss, grads = grad(model, loss_type="mi", clip_gradients=True)
        >>> optimizer.apply_gradients(zip(grads, model.trainable_variables))
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