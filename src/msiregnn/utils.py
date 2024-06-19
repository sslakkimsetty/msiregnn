""" Utility functions and QoL improvements. """

import msiregnn as msn
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

__all__ = [
    "checker_board",
    "grid",
    "simulate_theta_bspline",
    "imshow",
    "plot_vector_field"
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


def locnet_shape(
        img: tf.Tensor
) -> tuple[int, int]:
    """
    Evaluate the shape of the output of the locnet network.

    For a given input image, return the shape of the output feature map of the locnet network.

    :param img: the input image.
    :return: the shape of the output feature map of the locnet network.
    """
    return msn.LocNet()(img).numpy().shape


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
        model: msn.BsplineRegistration,
        fixed: tf.Tensor = tf.ones(shape=(1, 200, 200, 1)),
        moving: tf.Tensor = tf.ones(shape=(1, 200, 200, 1)),
        show: bool = True
) -> None:
    """
    Visualize coregistration images prior to and post coregistration.

    Plot four images in (1, 4) configuration containing the fixed image, the moving image,
    the transformed moving image post coregistration and the transformation applied on to a grid.

    :param model: the BsplineRegistration model.
    :param fixed: reference image (must be the same one used in the model training).
    :param moving: target image (must be the same one used in the model training).
    :param show: whether to show the plot immediately.

    :return: None (draws the composite plot in the plot pane).
    """
    (moving_hat, _), theta_hat = model(moving)
    stn = msn.SpatialTransformerBspline(img_res=model.img_res,
                                        grid_res=model.grid_res,
                                        out_dims=(model.img_res), B=1)
    fixed_hat, _ = stn(fixed, theta=theta_hat, B=1)

    plt.figure(figsize=(24,6))
    plt.subplot(141)
    imshow(fixed, title="fixed, "+str(fixed.shape), show=False)

    plt.subplot(142)
    imshow(moving, title="moving, "+str(moving.shape), show=False)

    plt.subplot(143)
    imshow(moving_hat, title="tr. moving, "+str(moving_hat.shape), show=False)

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

    plt.quiver(xs, ys, x_comp, y_comp)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    if show:
        plt.show()


def visualize_coreg_results(
        model: msn.BsplineRegistration,
        fixed: tf.Tensor,
        moving: tf.Tensor,
        loss_list: list[tf.Tensor],
        delta_true: tf.Tensor | None = None,
        delta_hat: tf.Tensor | None = None,
        theta_true: tf.Tensor | None = None,
        theta_hat: tf.Tensor | None = None
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
    msn.utils.visualize_coreg_images(model, fixed, moving)

    # Loss graph
    plt.figure(figsize=(6, 6))
    plt.plot(range(len(loss_list)), loss_list)
    plt.title("Loss")
    plt.xlabel("Iteration #")
    plt.ylabel("Loss")
    plt.show()

    # Dense vector field
    if delta_true is not None:
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        msn.utils.plot_vector_field(delta_true, title="True DVF", show=False)
        plt.subplot(122)
        msn.utils.plot_vector_field(delta_hat, title="Predicted DVF", show=True)
    else:
        plt.figure(figsize=(6, 6))
        msn.utils.plot_vector_field(delta_hat, title="Predicted DVF", show=True)

    # Control grid vector field
    if theta_true is not None:
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        msn.utils.plot_vector_field(theta_true, title="True control grid vector field", show=False)
        plt.subplot(122)
        msn.utils.plot_vector_field(theta_hat, title="Predicted control grid vector field")
    else:
        plt.figure(figsize=(6, 6))
        msn.utils.plot_vector_field(theta_hat, title="Predicted control grid vector field")



