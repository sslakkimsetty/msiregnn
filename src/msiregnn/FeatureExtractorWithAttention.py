"""Provides class definitions for FeatureExtractor layers for *Registration model."""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Resizing

# __all__ = [
#     "FeatureExtractor"
# ]

class FeatureExtractor(tf.keras.layers.Layer):
    def __init__(self, target_shape):
        super(FeatureExtractor, self).__init__()
        self.target_shape = target_shape
        self.layers_list = []

    def build(self, input_shape):
        current_shape = input_shape[1:3]
        target_shape = self.target_shape[1:3]

        while current_shape[0] > target_shape[0] or current_shape[1] > target_shape[1]:
            if current_shape[0] > target_shape[0]:
                self.layers_list.append(Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu'))
                current_shape = (current_shape[0], current_shape[1])
            if current_shape[1] > target_shape[1]:
                self.layers_list.append(Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu'))
                current_shape = (current_shape[0], current_shape[1])
            self.layers_list.append(MaxPooling2D(pool_size=2, strides=2, padding='same'))
            current_shape = (current_shape[0] // 2, current_shape[1] // 2)

        self.layers_list.append(Resizing(height=target_shape[0], width=target_shape[1]))
        self.layers_list.append(Conv2D(filters=self.target_shape[-1], kernel_size=1, activation='linear'))

    def call(self, inputs, masks):
        x = inputs
        for layer in self.layers_list:
            x = layer(x)
            mask_resized = tf.image.resize(masks, size=x.shape[1:3], method='bilinear')
            mask_resized = tf.expand_dims(mask_resized, axis=-1)
            x = tf.where(mask_resized > 0.5, x * mask_resized, tf.zeros_like(x))
            print(f"Shape after {layer.__class__.__name__}: {x.shape}")
        return x

    def compute_gradients(self, inputs, masks, loss_fn):
        with tf.GradientTape() as tape:
            predictions = self.call(inputs, masks)
            loss = loss_fn(inputs, predictions)
        gradients = tape.gradient(loss, self.trainable_variables)
        mask_resized = tf.image.resize(masks, size=gradients[0].shape[1:3], method='bilinear')
        mask_resized = tf.expand_dims(mask_resized, axis=-1)
        masked_gradients = [tf.where(mask_resized > 0.5, grad * mask_resized, tf.zeros_like(grad)) for grad in gradients]
        return masked_gradients, loss

        # Create an instance of the AttentionModel

    attention_model = AttentionModel()

    # Dummy data for training
    input_tensor = tf.random.uniform(shape=[1, 50, 258, 800])

    # Training loop (simplified)
    epochs = 10
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.MeanSquaredError()

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            predictions = attention_model(input_tensor)
            loss = loss_fn(input_tensor, predictions)

        gradients = tape.gradient(loss, attention_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, attention_model.trainable_variables))

        print(f"Epoch {epoch + 1}, Loss: {loss.numpy()}")

    # Generate soft attention masks after training
    attention_masks = attention_model.create_attention_masks(input_tensor)
    print(attention_masks.numpy())

    # Create an instance of the FeatureExtractor
    target_shape = (1, 13, 13, 2)
    feature_extractor = FeatureExtractor(target_shape=target_shape)

    # Compute gradients with masking
    gradients, loss = feature_extractor.compute_gradients(input_tensor, attention_masks, loss_fn)
    print(f"Loss: {loss.numpy()}")
