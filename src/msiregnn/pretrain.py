"""Provides class definitions for pretraining models for coregistration."""

import numpy as np
import tensorflow as tf


def pretrain_model(
        model,
        theta_id = None,
        epochs = 100,
        learning_rate = 0.001,
        strategy = "identity"  # New parameter
):
    if model.__class__.__name__ == "AffineRegistration":
        _pretrain_affine_model(
            model,
            theta_id = theta_id,
            epochs = epochs,
            learning_rate = learning_rate,
            strategy = strategy)  # Pass strategy
    elif model.__class__.__name__ == "BsplineRegistration":
        _pretrain_bspline_model(
            model,
            theta_id=theta_id,
            epochs=epochs,
            learning_rate=learning_rate)

def _pretrain_affine_model(
        model,
        theta_id = tf.constant(
            [
                [1, 0, 0],
                [0, 1, 0]], dtype = tf.float32
        ),
        epochs = 10,
        learning_rate = 0.001,
        strategy = "identity"
):
    """
    Pretrain affine model with different strategies.
    
    Args:
        model: AffineRegistration model
        theta_id: Target transformation for identity strategy
        epochs: Number of pretraining epochs
        learning_rate: Learning rate for pretraining
        strategy: One of "identity", "multi_transform", "curriculum", "self_supervised"
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.MeanSquaredError()

    if strategy == "identity":
        # Original behavior - train to output identity transformation
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                model()
                loss = loss_fn(theta_id, model.theta)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            if epoch % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.numpy()}")

    elif strategy == "multi_transform":
        # Train with multiple known transformations
        _pretrain_multi_transform(model, epochs, learning_rate)

    elif strategy == "curriculum":
        # Gradually increase transformation complexity
        _pretrain_curriculum(model, epochs, learning_rate)

    elif strategy == "self_supervised":
        # Learn to undo random transformations
        _pretrain_self_supervised(model, epochs, learning_rate)

    else:
        raise ValueError(f"Unknown pretraining strategy: {strategy}")

    print("Pretraining complete!")

def _pretrain_multi_transform(model, epochs, learning_rate):
    """Pretrain with multiple known transformations."""
    import msiregnn as msn
    from msiregnn.utils import affine_matrix_from_params

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Define a set of training transformations
    pretrain_configs = [
        {"scale_x": 1.0, "scale_y": 1.0, "angle": 0, "trans_x": 0, "trans_y": 0},
        {"scale_x": 1.05, "scale_y": 1.05, "angle": 0, "trans_x": 0, "trans_y": 0},
        {"scale_x": 0.95, "scale_y": 0.95, "angle": 0, "trans_x": 0, "trans_y": 0},
        {"scale_x": 1.0, "scale_y": 1.0, "angle": 5, "trans_x": 0, "trans_y": 0},
        {"scale_x": 1.0, "scale_y": 1.0, "angle": -5, "trans_x": 0, "trans_y": 0},
        {"scale_x": 1.0, "scale_y": 1.0, "angle": 0, "trans_x": 0.05, "trans_y": 0},
        {"scale_x": 1.0, "scale_y": 1.0, "angle": 0, "trans_x": -0.05, "trans_y": 0},
        {"scale_x": 1.0, "scale_y": 1.0, "angle": 0, "trans_x": 0, "trans_y": 0.05},
        {"scale_x": 1.0, "scale_y": 1.0, "angle": 0, "trans_x": 0, "trans_y": -0.05},
        {"scale_x": 1.02, "scale_y": 0.98, "angle": 3, "trans_x": 0.02, "trans_y": -0.02},
    ]

    for epoch in range(epochs):
        # Cycle through configurations
        config = pretrain_configs[epoch % len(pretrain_configs)]

        # Create target transformation
        theta_target = affine_matrix_from_params(**config)

        # Apply transformation to fixed image
        moving_synthetic = model.transformer(model.fixed, theta=theta_target, B=model.B)

        with tf.GradientTape() as tape:
            # Temporarily set synthetic as moving
            original_moving = model.moving
            model.moving = moving_synthetic

            # Forward pass
            model()

            # Parameter loss
            param_loss = tf.reduce_mean(tf.square(model.theta - theta_target))

            # Image similarity loss (negative MI)
            image_loss = -msn.metrics.mi(model.fixed, model.moving_hat)

            # Combined loss with more weight on parameters during pretraining
            total_loss = param_loss + 0.1 * image_loss

        # Update weights
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Restore original moving image
        model.moving = original_moving

        if epoch % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Param Loss: {param_loss.numpy():.4f}, "
                  f"Image Loss: {image_loss.numpy():.4f}")

def _pretrain_curriculum(model, epochs, learning_rate):
    """Curriculum learning - gradually increase transformation complexity."""
    import msiregnn as msn
    from msiregnn.utils import affine_matrix_from_params

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    for epoch in range(epochs):
        # Progress from 0 to 1
        progress = epoch / epochs

        # Gradually increase ranges
        scale_range = 0.0 + 0.15 * progress    # 0 to ±15%
        angle_range = 0.0 + 20.0 * progress    # 0 to ±20 degrees
        trans_range = 0.0 + 0.1 * progress     # 0 to ±0.1

        # Sample random transformation within current range
        scale_x = 1.0 + np.random.uniform(-scale_range, scale_range)
        scale_y = 1.0 + np.random.uniform(-scale_range, scale_range)
        angle = np.random.uniform(-angle_range, angle_range)
        trans_x = np.random.uniform(-trans_range, trans_range)
        trans_y = np.random.uniform(-trans_range, trans_range)

        # Create transformation
        theta_target = affine_matrix_from_params(
            scale_x=scale_x, scale_y=scale_y, angle=angle,
            trans_x=trans_x, trans_y=trans_y
        )

        # Apply to fixed image
        moving_synthetic = model.transformer(model.fixed, theta=theta_target, B=model.B)

        with tf.GradientTape() as tape:
            original_moving = model.moving
            model.moving = moving_synthetic

            model()

            # Loss computation
            param_loss = tf.reduce_mean(tf.square(model.theta - theta_target))
            image_loss = -msn.metrics.mi(model.fixed, model.moving_hat)
            total_loss = param_loss + 0.1 * image_loss

        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        model.moving = original_moving

        if epoch % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Progress: {progress:.2f}, "
                  f"Loss: {total_loss.numpy():.4f}")

def _pretrain_self_supervised(model, epochs, learning_rate):
    """Self-supervised pretraining by learning to undo transformations."""
    import msiregnn as msn
    from msiregnn.utils import affine_matrix_from_params

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    for epoch in range(epochs):
        # Generate random transformation
        scale = np.random.uniform(0.9, 1.1)
        angle = np.random.uniform(-15, 15)
        trans_x = np.random.uniform(-0.1, 0.1)
        trans_y = np.random.uniform(-0.1, 0.1)

        # Create transformation
        theta_forward = affine_matrix_from_params(
            scale_x=scale, scale_y=scale, angle=angle,
            trans_x=trans_x, trans_y=trans_y
        )

        # Transform fixed image
        augmented = model.transformer(model.fixed, theta=theta_forward, B=model.B)

        with tf.GradientTape() as tape:
            # Use augmented as moving image
            original_moving = model.moving
            model.moving = augmented

            model()

            # Goal: recover the original fixed image
            loss = -msn.metrics.mi(model.fixed, model.moving_hat)

            # Add small regularization
            if hasattr(model, 'theta_raw'):
                reg_loss = tf.reduce_sum(tf.square(model.theta_raw)) * 0.001
                total_loss = loss + reg_loss
            else:
                total_loss = loss

        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        model.moving = original_moving

        if epoch % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss.numpy():.4f}")
    print("Pretraining complete!")

def _pretrain_bspline_model(
        model,
        theta_id = None,
        epochs = 10,
        learning_rate = 0.001
):
    if theta_id is None:
        theta_id = tf.zeros(shape=model.grid_res, dtype=tf.float32)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.MeanSquaredError()

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            model()
            loss = loss_fn(theta_id, model.theta)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.numpy()}")

    print("Pretraining complete!")
