"""Testing grounds for interpolation methods."""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import msiregnn as msn
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Flatten,
    MaxPooling2D,
)

from PIL import Image
import os


img_res = (200, 180)
fixed = Image.open(os.path.join("/Users/lakkimsetty.s/Documents/",
                   "9-data/13-MUSC/",
                   "1-230206-liver/2-230206-liver-MALDI-first/",
                   "20230206_Liver_MALDI First_x2.5_z0.tif")).convert("L")
fixed = fixed.resize(img_res, Image.LANCZOS)
fixed = np.array(fixed, dtype=np.float32).T
fixed = remove_nans_tensor(fixed)
fixed = tf.expand_dims(fixed, axis=0)
fixed = tf.expand_dims(fixed, axis=-1)
fixed_min = tf.reduce_min(fixed)
fixed_max = tf.reduce_max(fixed)
fixed = (fixed - fixed_min) / (fixed_max - fixed_min)

theta = generate_affine_matrix(scale=1.05, angle=10, center=(0.5,0.5))
sta = msn.SpatialTransformerAffine(img_res=img_res, out_dims=img_res, B=1)
moving = sta(fixed, theta=theta, B=1)
moving_min = tf.reduce_min(moving)
moving_max = tf.reduce_max(moving)
moving = (moving - moving_min) / (moving_max - moving_min)
msn.utils.imshow(moving)


def apply_affine_transform(image, transform_matrix,
                           output_size, interpolation="BICUBIC",
                           fill_value=0):
    """
    Apply affine transformation using inverse mapping.

    Parameters:
        image: TensorFlow tensor of shape (H, W, C)
        transform_matrix: 2x3 affine transformation matrix
        output_size: Tuple (new_height, new_width)
        interpolation: Interpolation method ("BILINEAR" or "BICUBIC")

    Returns:
        Transformed image tensor
    """
    # Convert affine matrix to 1D tensor format (required by TF)
    transform_flattened = tf.concat([tf.reshape(transform_matrix, [-1]), tf.constant([0.0, 0.0])], axis=0)  # Shape (8,)
    transform_flattened = tf.reshape(transform_flattened, (1, 8))  # Shape (1, 8) for batch processing

    print(transform_flattened)

    # Apply inverse affine transformation
    transformed_image = tf.raw_ops.ImageProjectiveTransformV3(
        images=image,
        transforms=transform_flattened,
        output_shape=output_size,
        interpolation=interpolation,
        fill_value=fill_value
    )
    return transformed_image


moving_hat_2 = apply_affine_transform(image = moving,
                                      transform_matrix = theta_hat,
                                      output_size = img_res,
                                      interpolation = "BICUBIC")
msn.utils.imshow(moving_hat_2)
