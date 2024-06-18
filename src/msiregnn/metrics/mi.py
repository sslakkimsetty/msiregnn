"""Evaluates MI between two images that is differentiable."""

import numpy as np
import tensorflow as tf
from numpy.random import PCG64, Generator

__all__ = [
    "mi"
]


def sample_coords(
        dims: tuple[int, int],
        n: int
) -> tuple[np.array, np.array]:
    """
    Sample random coordinates within the specified dimensions.

    :param dims: Dimensions of the space in the format (H, W).
    :param n: Number of random coordinates to sample.

    :return: Tuple containing two arrays of random y and x coordinates.
    """
    height, width = dims  # dims are (H,W)

    rng = Generator(PCG64())
    ix = rng.choice(width, size=n)
    iy = rng.choice(height, size=n)
    return (iy, ix)


def Gphi( # noqa
        z: tf.Tensor,
        phi: tf.Tensor,
        _type: str = "marginal"
) -> tf.Tensor:
    """
    Evaluate the Gaussian density function for given inputs.

    This function calculates the Gaussian density function based on the provided inputs.

    :param z: Input data tensor.
    :param phi: Covariance matrix or variance, depending on the type.
    :param _type: Type of the Gaussian distribution. Options are "marginal" or "joint".

    :return: Tensor containing the evaluated Gaussian density function.
    """
    # if type is "joint", z is expected in nx2 shape
    # n isequalto len(z)
    if _type == "marginal":
        phi_det = phi
        c = (-1 / 2) * ((z ** 2) / phi)
        k = 1
    else:
        phi_det = tf.linalg.det(phi)
        _a = tf.linalg.inv(phi)
        _b = tf.matmul(z, _a)
        _d = _b * z
        c = (-1 / 2) * (tf.reduce_sum(_d, axis=1))
        k = 2

    a = (2 * np.pi) ** (-k / 2)
    b = phi_det ** (-1 / 2)
    return a * b * tf.exp(c)


def construct_z(
        img: tf.Tensor,
        c: tuple[np.array, np.array, np.array, np.array]
) -> tf.Tensor:
    """
    Construct the difference vector z based on image and coordinates.

    This function constructs the difference vector z using the image and provided coordinates.

    :param img: Input image tensor.
    :param c: Coordinates for constructing the difference vector (cix, ciy, cjx, cjy).

    :return: Flattened difference vector z.
    """
    cix, ciy, cjx, cjy = c
    n = len(cix)

    zi = tf.gather_nd(img, np.vstack([ciy, cix]).T)
    zj = tf.gather_nd(img, np.vstack([cjy, cjx]).T)

    zi = tf.reshape(tf.tile(zi, [n]), (n, -1))
    zj = tf.reshape(tf.tile(zj, [n]), (-1, n))
    zj = tf.transpose(zj)

    z = zi - zj
    return tf.reshape(z, (-1,))


def _entropy(
        z: tf.Tensor,
        n: int,
        _type: str = "marginal",
        phi: float = 0.1
) -> tf.Tensor:
    """
    Compute the entropy of the given vector z.

    This function computes the entropy of the given vector z using the Gphi function.

    :param z: Input vector for entropy computation.
    :param n: Number of elements in the vector z.
    :param _type: Type of entropy calculation ("marginal" or "joint").
    :param phi: Precision parameter for Gphi function.

    :return: Entropy value.
    """
    g = Gphi(z, phi=phi, _type=_type)
    out = tf.reshape(g, (n, -1))
    out = (1 / n) * tf.reduce_sum(out, axis=1)
    out = tf.math.log(out)
    out = -(1 / n) * tf.reduce_sum(out)
    return out


def _compute_scale(
        z: np.ndarray
) -> float:
    """
    Compute the scale (standard deviation) of the given vector z.

    This function computes the scale (standard deviation) of the given vector z.

    :param z: Input vector for scale computation.

    :return: Scale (standard deviation) value.
    """
    return np.sqrt(np.var(z))


def mi(
        u: tf.Tensor,
        v: tf.Tensor,
        n: int = 100
) -> float:
    """
    Compute the mutual information between two images u and v using sampled coordinates.

    This function computes the mutual information between two images u and v using
    sampled coordinates.

    :param u: First input image.
    :param v: Second input image.
    :param n: Number of sampled coordinates for mutual information calculation. Default is 100.

    :return: Mutual information value.
    """
    u = tf.squeeze(u)
    v = tf.squeeze(v)
    height, width = u.shape
    dims = (height, width)

    # Sample coordinates for sample B
    ciy, cix = sample_coords(dims, n=n)

    # Sample coordinates for sample A
    cjy, cjx = sample_coords(dims, n=n)

    c = (cix, ciy, cjx, cjy)

    # Construct z for u and v
    uz = construct_z(u, c)
    vz = construct_z(v, c)

    n = len(cix)

    phi = np.average([_compute_scale(uz),
                      _compute_scale(vz)])
    phi = np.sqrt(0.1)
    sigma = np.eye(2) * phi

    # Entropy for u and v
    hu = _entropy(uz, n, phi=phi)
    hv = _entropy(vz, n, phi=phi)

    # Joint entropy
    uvz = tf.stack([uz, vz])
    uvz = tf.transpose(uvz)
    huv = _entropy(uvz, n, _type="joint", phi=tf.eye(2, 2) * phi)

    _mi = hu + hv - huv
    return _mi
