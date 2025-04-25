import tensorflow as tf
import numpy as np
from PIL import Image
import os
from .reg.AffineRegistration import AffineRegistration
from .stn.affine.st_affine import SpatialTransformerAffine
from .utils import generate_affine_matrix, nans_to_zeros

# Load and preprocess the fixed image
img_res = (200, 180)
fixed = Image.open(os.path.join("/Users/lakkimsetty.s/Documents/",
                   "9-data/13-MUSC/",
                   "1-230206-liver/2-230206-liver-MALDI-first/",
                   "20230206_Liver_MALDI First_x2.5_z0.tif")).convert("L")
# fixed = Image.open(os.path.join("/Users/sai/Documents/00-NEU/2-Ind-Study",
#                    "9-data/13-MUSC/",
#                    "1-230206-liver/2-230206-liver-MALDI-first/",
#                    "20230206_Liver_MALDI First_x2.5_z0.tif")).convert("L")
fixed = fixed.resize(img_res, Image.LANCZOS)
fixed = np.array(fixed, dtype=np.float32).T
fixed = nans_to_zeros(fixed)
fixed = tf.expand_dims(fixed, axis=0)
fixed = tf.expand_dims(fixed, axis=-1)
fixed_min = tf.reduce_min(fixed)
fixed_max = tf.reduce_max(fixed)
fixed = (fixed - fixed_min) / (fixed_max - fixed_min)

# Generate the moving image
theta = affine_matrix_from_params(scale=1.05, angle=10, center=(0.5, 0.5))
sta = SpatialTransformerAffine(img_res=img_res, out_dims=img_res, B=1)
moving = sta(fixed, theta=theta, B=1)
moving_min = tf.reduce_min(moving)
moving_max = tf.reduce_max(moving)
moving = (moving - moving_min) / (moving_max - moving_min)

# Instantiate the model
model = AffineRegistration(
    fixed = fixed,
    moving = moving,
    factor = 1,
    pretrain = True,
    theta_id = None,
    pretrain_epochs = 300,
    pretrain_lr = 0.001,
    regularize = True,
    reg_weight = 1e-3)

# Define the optimizer
optim = tf.keras.optimizers.Adam(learning_rate=5e-4)

# Train the model
ITERMAX = 100
model.train(optim = optim, ITERMAX = ITERMAX)