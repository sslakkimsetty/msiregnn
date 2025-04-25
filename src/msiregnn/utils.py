"""Utility functions and QoL improvements."""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from . import SpatialTransformerBspline

__all__ = [
    "checker_board",
    "grid",
    "simulate_theta_bspline",
    "imshow",
    "plot_vector_field",
    "locnet_output_shape"
]

def checker_board(
        shape: tuple[int, int] = (200, 200),
        res: int = 20
) -> tf.Tensor:
    """
    Create a Tensorflow Tensor in a checkerboard style format.

    :param shape: the dimensions (x and y) of the returned Tensor
    :param res: the number of squares (black and white)

    :return: a 4D Tensor of the size :param shape: with two added dimensions on either side
    """
    height, width = shape
    arr = (np.indices(shape) // res).sum(axis=0) % 2
    arr = tf.reshape(arr, (1, height, width, 1))
    arr = tf.cast(arr, tf.float32)
    return arr


def grid(
        shape: tuple[int, int] = (200, 200),
        res: int | tuple[int, int] | None = None,
        thickness: int | tuple[int, int] | None = None
) -> tf.Tensor:
    """
    Create a Tensorflow Tensor in a grid style format.

    :param shape: the dimensions (height and width) of the returned Tensor
    :param res: resolution of the grid (a white line every :param res: pixels).
    Defaults to one-tenth of the :param shape: if None.

    :param thickness: the width line forming the grid (a white line every :param linewidth: pixels).
    Defaults to one-hundredth of the :param shape: if None.

    :return: a 4D Tensor of the size :param shape: with two added dimensions on either side
    """
    height, width = shape

    if res is None:
        res = (np.floor_divide(height, 10), np.floor_divide(width, 10))
    elif type(res) is int:
        res = (res, res)

    if thickness is None:
        thickness = (np.floor_divide(height, 200), np.floor_divide(width, 200))
    elif type(thickness) is int:
        thickness = (thickness, thickness)

    arr = np.zeros(shape=shape)
    for ind in range(thickness[0]):
        arr[list(range(height))[ind::res[0]], :] = 1.0

    for ind in range(thickness[1]):
        arr[:, list(range(height))[ind::res[1]]] = 1.0

    arr = tf.reshape(arr, (1, height, width, 1))
    arr = tf.cast(arr, tf.float32)
    return arr


def simulate_theta_bspline(
        img_res: tuple[int, int],
        grid_res: tuple[int, int],
        std: float = 2.0
) -> tf.Tensor:
    """
    Create the transformation parameters of size :param grid_res:

    Create the transformation parameters of the control grid of the size :param img_res:. These
    parameters are used to evaluate the x- and y-displacements for each location (or pixel).

    :param img_res: the resolution of the input image.
    :param grid_res: the resolution in pixels between the control grid points.
    :param std: the standard deviation of the normal distribution used to simulate the
    transformation parameters.

    :return: a Tensor containing the transformation parameters.
    """
    height, width = img_res
    sx, sy = grid_res
    gx, gy = np.ceil(width / sx), np.ceil(height / sy)
    nx, ny = gx + 3, gy + 3
    nx, ny = nx.astype("int32"), ny.astype("int32")

    theta_x = np.random.normal(0.0, std, (ny, nx))
    theta_y = np.random.normal(0.0, std, (ny, nx))

    theta = tf.stack([theta_x, theta_y], axis=0)
    return theta


def imshow(
        inp: tf.Tensor | np.ndarray,
        figsize: tuple[int, int] | None = None,
        xlab: str | None = None,
        ylab: str | None = None,
        title: str | None = None,
        show: bool = True
) -> None:
    """
    Plot a Tensor as an image in "binary_r" cmap.

    :param inp: the input image to be plotted.
    :param figsize: the size of the figure.
    :param xlab: the label of the x axis.
    :param ylab: the label of the y axis.
    :param title: the title of the plot.
    :param show: whether to show the plot.

    :return: None
    """
    if figsize is not None:
        plt.figure(figsize=figsize)

    try:
        plt.imshow(inp.squeeze(), plt.cm.binary_r)
    except: # noqa
        plt.imshow(inp.numpy().squeeze(), plt.cm.binary_r)

    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    if show:
        plt.show()


def visualize_coreg_images(
        model,
        show: bool = True
) -> None:
    """
    Visualize coregistration images prior to and post coregistration.

    Plot four images in (1, 4) configuration containing the fixed image, the moving image,
    the transformed moving image post coregistration and the transformation applied on to a grid.

    :param model: the BsplineRegistration model.
    :param show: whether to show the plot immediately.

    :return: None (draws the composite plot in the plot pane).
    """
    stn = SpatialTransformerBspline(
        img_res = (model.img_res[1], model.img_res[2]),
        grid_res = model.grid_res,
        out_dims = (model.img_res[1], model.img_res[2]),
        B=1
    )
    fixed_hat, _ = stn(model.fixed, theta=model.theta, B=1)

    plt.figure(figsize=(24,6))
    plt.subplot(141)
    imshow(model.fixed, title="fixed, "+str(model.fixed.shape), show=False)

    plt.subplot(142)
    imshow(model.moving, title="moving, "+str(model.moving.shape), show=False)

    plt.subplot(143)
    imshow(model.moving_hat, title="tr. moving, "+str(model.moving_hat.shape), show=False)

    plt.subplot(144)
    imshow(fixed_hat, title="tr. grid, "+str(fixed_hat.shape), show=False)

    if show:
        plt.show()


def plot_vector_field(
        delta: tf.Tensor = tf.ones(shape=(2, 1, 200, 200)),
        title: str = "Dense vector field",
        show: bool = True
) -> None:
    """
    Plot the vector field with x- and y-components

    :param delta: the vector field as a Tensor
    :param title: the title of the plot
    :param show: whether to show the plot immediately

    :return: None
    """
    delta = delta.numpy().squeeze()
    y_comp = delta[0]
    x_comp = delta[1]

    meshsize = x_comp.shape
    y = np.arange(meshsize[0])
    x = np.arange(meshsize[1])
    xs, ys = np.meshgrid(x, y)

    plt.quiver(xs, ys, x_comp, y_comp, scale=1, scale_units="xy")
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    if show:
        plt.show()


def visualize_coreg_results(
        model,
        delta_true: tf.Tensor | None = None,
        theta_true: tf.Tensor | None = None,
) -> None:
    """
    Visualize the coregistration results with BsplineRegistration.

    :param model: the BsplineRegistration model.
    :param fixed: reference image (must be the same one used in the model training).
    :param moving: target image (must be the same one used in the model training).
    :param loss_list: list of losses computed during training.
    :param delta_true: Dense vector field of the ground truth transformation.
    :param delta_hat: Dense vector field of the predicted transformation.
    :param theta_true: Vector field of the control grid of the ground truth transformation.
    :param theta_hat: Vector field of the control grid of the predicted transformation.

    :return: None (draws several plots in the plot pane).
    """
    # Visualization of images prior to and post coregistration
    visualize_coreg_images(model)

    # Loss graph
    plt.figure(figsize=(6, 6))
    plt.plot(range(len(model.loss_list)), model.loss_list)
    plt.title("Loss")
    plt.xlabel("Iteration #")
    plt.ylabel("Loss")
    plt.show()

    # Dense vector field
    if delta_true is not None:
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plot_vector_field(delta_true, title="True DVF", show=False)
        plt.subplot(122)
        plot_vector_field(model.delta_hat, title="Predicted DVF", show=True)
    else:
        plt.figure(figsize=(6, 6))
        plot_vector_field(model.delta_hat, title="Predicted DVF", show=True)

    # Control grid vector field
    if theta_true is not None:
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plot_vector_field(theta_true, title="True control grid vector field", show=False)
        plt.subplot(122)
        plot_vector_field(model.theta, title="Predicted control grid vector field")
    else:
        plt.figure(figsize=(6, 6))
        plot_vector_field(model.theta, title="Predicted control grid vector field")


def locnet_output_shape(locnet, input_shape):
    """
    Evaluate the shape of the output of the locnet network.

    For a given input image, return the shape of the output feature map of the locnet network.

    :param loc_net: the locnet network.
    :param input_shape: the input image shape.

    :return: the shape of the output feature map of the locnet network.
    """
    dummy_input = tf.random.uniform(input_shape)  # Create a dummy input tensor with random values
    output = locnet(dummy_input)  # Perform a forward pass
    return output.shape


def nans_to_zeros(tensor):
    return tf.where(tf.math.is_nan(tensor), tf.zeros_like(tensor), tensor)


def affine_matrix_from_params(scale=1.0, angle=0, center=(0, 0)):
    scale = tf.constant(scale, dtype=tf.float32)
    theta = tf.experimental.numpy.deg2rad(tf.constant(angle, dtype=tf.float32))
    cx, cy = center

    translate_POT = tf.constant([
        [1, 0, -cx],
        [0, 1, -cy],
        [0, 0, 1]], dtype=tf.float32)

    a = scale * tf.math.cos(theta)
    b = scale * tf.math.sin(theta)

    sr_matrix = tf.stack([
        [a, -b, 0],
        [b, a, 0],
        [0, 0, 1]], axis=0)

    translate_POT_back = tf.constant([
        [1, 0, cx],
        [0, 1, cy],
        [0, 0, 1]], dtype=tf.float32)

    M = tf.linalg.matmul(translate_POT_back, tf.linalg.matmul(sr_matrix, translate_POT))
    return M[:2, :]

