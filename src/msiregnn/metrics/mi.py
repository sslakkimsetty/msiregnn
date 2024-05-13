"""
Evaluates MI between two images that is differentiable.
"""

import numpy as np
import tensorflow as tf
from numpy.random import PCG64, Generator

__all__ = [
    "mi"
]


def sample_coords(
        dims: tuple[int, int],
        n: int
) -> tuple[int, int]:
    """
    Sample random coordinates within the specified dimensions.

    :param dims: Dimensions of the space in the format (H, W).
    :param n: Number of random coordinates to sample.

    :return: Tuple containing two arrays of random y and x coordinates.
    """
    H, W = dims  # dims are (H,W)

    rng = Generator(PCG64())
    ix = rng.choice(W, size=n)
    iy = rng.choice(H, size=n)

    return (iy, ix)


def Gphi(z, phi, _type="marginal"):  # noqa: N802
    """
    Evaluate the Gaussian density function for given inputs.

    This function calculates the Gaussian density function based on the provided inputs.

    :param z: Input data tensor.
    :type z: tf.Tensor

    :param phi: Covariance matrix or variance, depending on the type.
    :type phi: tf.Tensor

    :param _type: Type of the Gaussian distribution. Options are "marginal" or "joint".
    :type _type: str, optional

    :return: Tensor containing the evaluated Gaussian density function.
    :rtype: tf.Tensor
    """
    # if type is "joint", z is expected in nx2 shape
    # n = len(z)
    if _type == "marginal":
        phi_det = phi
        C = (-1 / 2) * ((z ** 2) / phi)
        k = 1
    else:
        phi_det = tf.linalg.det(phi)
        _A = tf.linalg.inv(phi)
        _B = tf.matmul(z, _A)
        _D = _B * z
        C = (-1 / 2) * (tf.reduce_sum(_D, axis=1))
        k = 2

    A = (2 * np.pi) ** (-k / 2)
    B = phi_det ** (-1 / 2)
    return A * B * tf.exp(C)


def construct_z(img, c):
    """
    Construct the difference vector z based on image and coordinates.

    This function constructs the difference vector z using the image and provided coordinates.

    :param img: Input image tensor.
    :type img: tf.Tensor

    :param c: Coordinates for constructing the difference vector (cix, ciy, cjx, cjy).
    :type c: tuple

    :return: Flattened difference vector z.
    :rtype: tf.Tensor
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


def _entropy(z, n, _type="marginal", phi=0.1):
    """
    Compute the entropy of the given vector z.

    This function computes the entropy of the given vector z using the Gphi function.

    :param z: Input vector for entropy computation.
    :type z: tf.Tensor

    :param n: Number of elements in the vector z.
    :type n: int

    :param _type: Type of entropy calculation ("marginal" or "joint").
    :type _type: str, optional

    :param phi: Precision parameter for Gphi function.
    :type phi: float, optional

    :return: Entropy value.
    :rtype: tf.Tensor
    """
    g = Gphi(z, phi=phi, _type=_type)
    out = tf.reshape(g, (n, -1))
    out = (1 / n) * tf.reduce_sum(out, axis=1)
    out = tf.math.log(out)
    out = -(1 / n) * tf.reduce_sum(out)
    return out


def _compute_scale(z):
    """
    Compute the scale (standard deviation) of the given vector z.

    This function computes the scale (standard deviation) of the given vector z.

    :param z: Input vector for scale computation.
    :type z: np.ndarray

    :return: Scale (standard deviation) value.
    :rtype: float
    """
    return np.sqrt(np.var(z))


def mi(u, v, n=100):
    """
    Compute the mutual information between two images u and v using sampled coordinates.

    This function computes the mutual information between two images u and v using
    sampled coordinates.

    :param u: First input image.
    :type u: tf.Tensor
    :param v: Second input image.
    :type v: tf.Tensor
    :param n: Number of sampled coordinates for mutual information calculation. Default is 100.
    :type n: int

    :return: Mutual information value.
    :rtype: float
    """
    u = tf.squeeze(u)
    v = tf.squeeze(v)
    H, W = u.shape
    dims = (H, W)

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

    # print("hu:", hu.numpy())
    # print("hv:", hv.numpy())

    # Joint entropy
    uvz = tf.stack([uz, vz])
    uvz = tf.transpose(uvz)
    huv = _entropy(uvz, n, _type="joint", phi=tf.eye(2, 2) * phi)
    # print("huv:", huv.numpy())

    _mi = hu + hv - huv
    return _mi
