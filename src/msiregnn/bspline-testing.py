"""Testing grounds for new methods."""

import tensorflow as tf
import numpy as np
import msiregnn as msn
from src.msiregnn.reg import LocNet, BsplineRegistration
from src.msiregnn import SpatialTransformerBspline
from src.msiregnn.utils import (checker_board, simulate_theta_bspline,
                                imshow, nans_to_zeros)

from PIL import Image
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="mpld3")
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")


################################
##### UTILITY FUNCTIONS
################################


################################
##### BSPLINE REGISTRATION
################################

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
grid_res_true = (10, 10)
theta_true = simulate_theta_bspline(img_res, grid_res_true, std=1.0)
grid_res = (theta_true.shape[1], theta_true.shape[2])
stb = SpatialTransformerBspline(img_res=img_res, grid_res=grid_res, B=1)
moving, delta_true = stb(fixed, theta=theta_true, B=1)
moving_min = tf.reduce_min(moving)
moving_max = tf.reduce_max(moving)
moving = (moving - moving_min) / (moving_max - moving_min)

# Instantiate the model
model = BsplineRegistration(
    fixed = fixed,
    moving = moving,
    pretrain = True,
    regularize = True,
    factor = 1,
    pretrain_epochs = 25)

# Define the optimizer
optim = tf.keras.optimizers.Adam(learning_rate=5e-3)

# Train the model
ITERMAX = 1000
model.train(optim = optim, ITERMAX = ITERMAX, loss_type="mse")

# Visualize the coregistration results with BsplineRegistration
msn.utils.visualize_coreg_results(model, delta_true, theta_true)




