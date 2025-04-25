"""Provides class definitions for pretraining models for coregistration."""

import tensorflow as tf

def pretrain_model(
        model,
        theta_id = None,
        epochs = 100,
        learning_rate = 0.001
):
    if model.__class__.__name__ == "AffineRegistration":
        _pretrain_affine_model(
            model,
            theta_id = theta_id,
            epochs = epochs,
            learning_rate = learning_rate)
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
        learning_rate = 0.001
):
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