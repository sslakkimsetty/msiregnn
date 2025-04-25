"""Testing grounds for new methods."""

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


################################
##### UTILITY FUNCTIONS
################################


################################
##### BSPLINE REGISTRATION
################################

class LocNet(tf.keras.layers.Layer):

    def __init__(self):
        super(LocNet, self).__init__()

        self.conv1 = Conv2D(filters=32, kernel_size=5, activation="relu", strides=2)
        self.avgp1 = MaxPooling2D(pool_size=(2,2), strides=(1,1))
        self.conv2 = Conv2D(filters=32, kernel_size=5, activation="relu", strides=2)
        self.avgp2 = MaxPooling2D(pool_size=(2,2), strides=(1,1))
        self.conv3 = Conv2D(filters=32, kernel_size=5, activation="relu", strides=1)
        self.avgp3 = MaxPooling2D(pool_size=(2,2), strides=(1,1))
        self.conv4 = Conv2D(filters=32, kernel_size=5, activation="relu", strides=1)
        self.avgp4 = MaxPooling2D(pool_size=(2,2), strides=(1,1))
        self.conv5 = Conv2D(filters=32, kernel_size=5, activation="relu", strides=2)
        self.conv6 = Conv2D(filters=32, kernel_size=5, activation="relu", strides=1)
        self.conv7 = Conv2D(filters=32, kernel_size=1, activation=None, strides=1)
        self.conv8 = Conv2D(filters=2, kernel_size=1, activation="sigmoid", strides=1)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.avgp1(x)
        x = self.conv2(x)
        x = self.avgp2(x)
        # x = self.conv3(x)
        # x = self.avgp3(x)
        # x = self.conv4(x)
        # x = self.avgp4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        return x


################################
##### BSPLINE TESTING
################################

img_res = (200, 180)
res = 20
grid_res_true = (10, 10)
std = 3.0

cb1 = checker_board(shape=img_res, res=res)
imshow(cb1)
theta_true = simulate_theta_bspline(img_res, grid_res_true, std=3.0)
print(theta_true.shape)

grid_res = (theta_true.shape[1], theta_true.shape[2])
stb = SpatialTransformerBspline(img_res=img_res, grid_res=grid_res, B=1)
out, delta_true = stb(cb1, theta=theta_true, B=1)

imshow(out)

fixed = cb1
moving = out

grid_res = msn.utils.locnet_output_shape(fixed, loc_net=LocNet())
grid_res = (grid_res[1], grid_res[2])
print(grid_res)

loc_net = LocNet()
model = msn.BsplineRegistration(img_res=img_res,
                                loc_net=loc_net, factor=1)
optim = tf.keras.optimizers.legacy.Adagrad(learning_rate=5e-1)
model, loss_list = msn.training.train(model, fixed, moving,
                                      optim=optim, ITERMAX=2000)
(moving_hat, delta_hat), theta_hat = model(moving)

# Visualize the coregistration results with BsplineRegistration
msn.utils.visualize_coreg_results(model, fixed, moving,
                                  loss_list, delta_true,
                                  delta_hat, theta_true, theta_hat)


# Test simulated data at real data's resolution
delta_true_piece = delta_true[:, :, 100:350, 100:350]
delta_hat_piece = delta_hat[:, :, 100:350, 100:350]

plt.figure(figsize=(40, 20))
plt.subplot(121)
msn.utils.plot_vector_field(delta_true_piece, show=False)
plt.subplot(122)
msn.utils.plot_vector_field(delta_hat_piece)


# Test real data
# fixed = np.loadtxt("/Users/lakkimsetty.s/Library/Mobile Documents/com~apple~CloudDocs/"
#                    "01-NU/tonsil-fixed.csv", delimiter=",")
# moving = np.loadtxt("/Users/lakkimsetty.s/Library/Mobile Documents/com~apple~CloudDocs/"
#                     "01-NU/tonsil-moving.csv", delimiter=",")

fixed = np.loadtxt("/Users/sai/Documents/"
                   "01-NU/tonsil-fixed.csv", delimiter=",")
moving = np.loadtxt("/Users/sai/Documents/"
                    "01-NU/tonsil-moving.csv", delimiter=",")

resize_ratio = 4
fshape = tuple([round(x / resize_ratio) for x in fixed.shape])
mshape = tuple([round(x / resize_ratio) for x in moving.shape])

from PIL import Image

fixed = Image.fromarray(fixed)
moving = Image.fromarray(moving)

fixed = fixed.resize(size=fshape)
moving = moving.resize(size=mshape)

fixed = np.asarray(fixed)
moving = np.asarray(moving)

fixed = tf.expand_dims(fixed, axis=0)
fixed = tf.expand_dims(fixed, axis=3)
moving = tf.expand_dims(moving, axis=0)
moving = tf.expand_dims(moving, axis=3)

fshape = fixed.shape
mshape = moving.shape
img_res = fshape[1:3]
















