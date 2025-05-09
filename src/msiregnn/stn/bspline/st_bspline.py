"""Provides class definition and methods for SpatialTransformerBspline."""

import tensorflow as tf

__all__ = [
    "SpatialTransformerBspline"
]

class SpatialTransformerBspline(tf.keras.layers.Layer):
    """
    Spatial Transformer Bspline layer for spatial transformations using bspline functions.

    This layer applies spatial transformations on input feature maps based on B-spline
    transformation parameters. It consists of a grid generator and a bilinear sampler for
    performing differentiable spatial transformations using B-spline basis functions.
    """

    def __init__(
            self,
            img_res: tuple[int, int] = (100, 100),
            grid_res: tuple[int, int] | None = None,
            out_dims: tuple[int, int] | None = None,
            B: int = 1
    ):
        """
        Class initializer for the SpatialTransformerBspline class.

        :param img_res: Resolution of the input images (H, W).
        Defaults to (100, 100) if not provided.

        :param grid_res: Resolution of the control grid points (nx, ny).
        Defaults to (ceil(W/7), ceil(H/7)) if not provided.

        :param out_dims: Output dimensions of the transformed images (out_H, out_W).
        Defaults to `img_res` if not provided.

        :param B: Batch size. Defaults to None.

        :Attributes:
            - H (int): Height of the input images.
            - W (int): Width of the input images.
            - out_H (int): Height of the output images.
            - out_W (int): Width of the output images.
            - nx (int): Number of control points in the x-direction.
            - ny (int): Number of control points in the y-direction.
            - gx (tf.Tensor): Grid size in the x-direction.
            - gy (tf.Tensor): Grid size in the y-direction.
            - sx (tf.Tensor): Scaling factor in the x-direction.
            - sy (tf.Tensor): Scaling factor in the y-direction.
            - base_grid (tf.Tensor): Base grid for the grid generator.
            - px (tf.Tensor): Base indices for B-spline basis functions in the x-direction.
            - py (tf.Tensor): Base indices for B-spline basis functions in the y-direction.
            - Buv (tf.Tensor): B-spline basis functions.

        :Methods:
            - _transformer(input_fmap, theta=None): Applies the spatial transformation on the input feature map.
            - _grid_generator(theta=None): Generates the grid for B-spline transformation.
            - _delta_calculator(b, px, py, theta): Computes the delta values for B-spline transformation.
            - _compute_theta_slices(b, px, py, theta, i): Computes theta slices for B-spline transformation.
            - _piece_bsplines(u): Computes piece-wise B-spline basis functions.
            - _bilinear_sampler(img, x, y): Bilinear sampler for sampling values from the input feature map.
            - _pixel_intensity(img, x, y): Retrieves pixel intensities from the input feature map.
            - call(input_fmap, theta=None, B=None): Applies the spatial transformer B-spline layer on the input feature map.

        :Example usage:
        ```python
        st_bspline_layer = SpatialTransformerBspline(img_res=(256, 256), grid_res=(32, 32), out_dims=(128, 128), B=32)
        input_feature_map = tf.random.normal(shape=(32, 256, 256, 3))
        theta_params = tf.random.normal(shape=(32, 2, 32, 32))  # Example B-spline transformation parameters
        transformed_output, delta = st_bspline_layer.call(input_feature_map, theta=theta_params, B=32)
        ```

        :Note: This layer is designed to be used in neural network architectures
        for tasks involving spatial transformations using B-spline functions.
        """
        super().__init__()

        # !!! TODO support for multi channel featuremaps is missing!

        #### MAIN TRANSFORMER FUNCTION PART ####
        if not img_res:
            img_res = (100, 100)
        self.H, self.W = img_res

        if not out_dims:
            out_dims = img_res
        self.out_H, self.out_W = out_dims

        if not grid_res:
            grid_res = (tf.math.ceil(self.W/7), tf.math.ceil(self.Y/7)) #$ #$

        ny, nx = grid_res #$
        self.nx, self.ny = tf.cast(nx, tf.int32), tf.cast(ny, tf.int32)

        gx, gy = self.nx-3, self.ny-3
        sx, sy = tf.cast(self.W/gx, tf.float32), tf.cast(self.H/gy, tf.float32) #$ #$


        #### GRID GENERATOR PART ####
        # Create grid
        x = tf.linspace(start=0.0, stop=self.W-1, num=self.W) #$ #$
        y = tf.linspace(start=0.0, stop=self.H-1, num=self.H) #$ #$
        xt, yt = tf.meshgrid(x, y)

        xt = tf.expand_dims(xt, axis=0)
        xt = tf.tile(xt, tf.stack([B,1,1]))

        yt = tf.expand_dims(yt, axis=0)
        yt = tf.tile(yt, tf.stack([B,1,1]))

        self.base_grid = tf.stack([xt, yt], axis=0) #$ #$

        # Calculate base indices and piece wise bspline function inputs
        self.px, self.py = tf.floor(xt/sx), tf.floor(yt/sy)
        u = (xt/sx) - self.px
        v = (yt/sy) - self.py

        self.px = tf.cast(self.px, tf.int32)
        self.py = tf.cast(self.py, tf.int32)

        # Compute Bsplines
        # Bu and Bv have shapes (B*H*W, 4)
        Bu = self._piece_bsplines(u) ##
        Bu = tf.reshape(Bu, shape=(4,-1))
        Bu = tf.transpose(Bu)
        Bu = tf.reshape(Bu, (-1,4,1))

        Bv = self._piece_bsplines(v) ##
        Bv = tf.reshape(Bv, shape=(4,-1))
        Bv = tf.transpose(Bv)
        Bv = tf.reshape(Bv, (-1,1,4))

        self.Buv = tf.matmul(Bu,Bv)
        self.Buv = tf.reshape(self.Buv, (B,self.H,self.W,4,4))


    def _transformer(
            self,
            input_fmap: tf.Tensor,
            theta: tf.Tensor = None
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Apply the spatial transformation on the input feature map using Bspline transformations.

        This method transforms the input feature map based on the given Bspline
        transformation parameters (`theta`).

        :param input_fmap: input feature map with shape (B, H, W, C).
        :param theta: bspline transformation parameters with shape (B, 2, ny, nx). Defaults to None.

        :return: a tuple consisting of two Tensors,
            - out_fmap, Transformed output feature map with shape (B, out_H, out_W, C).
            - delta, Transformation deltas.

        :Note: This method is used internally by the SpatialTransformerBspline layer.
        """
        B, H, W, C = input_fmap.shape
        if B is None:
            self.B = 1

        # Initialize theta to identity transformation if not provided
        if type(theta) == type(None):
            theta = tf.zeros((self.B,2,self.ny,self.nx), tf.float32)
        else:
            try:
                theta = tf.reshape(theta, shape=[self.B,2,self.ny,self.nx])
            except:
                theta = tf.reshape(theta, shape=[-1,2,self.ny,self.nx])
                self.B = theta.shape[0]

        batch_grids, delta = self._grid_generator(theta)

        # Extract source coordinates
        # batch_grids has shape (2,B,H,W)
        xs = batch_grids[0, :, :, :]
        ys = batch_grids[1, :, :, :]

        # Compile output feature map
        out_fmap = self._bilinear_sampler(input_fmap, xs, ys) ##
        return out_fmap, delta


    def _grid_generator(
            self,
            theta: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Generate the control grid and compute transformation deltas.

        This method generates the control grid based on the given Bspline transformation
        parameters (theta) and computes the transformation deltas.

        :param theta: Bspline transformation parameters with shape (B, 2, ny, nx).
        Defaults to None.

        :return: A tuple of tensors containing,
            - batch_grids, generated batch grids with shape (2, B, H, W).
            - delta, Transformation deltas.

        :Note: This method is used internally by the SpatialTransformerBspline layer.
        """
        # theta shape B, 2, ny, nx
        theta_x = theta[:,0,:,:]
        theta_y = theta[:,1,:,:]

        px = self.px
        py = self.py

        batch_idx = tf.range(0, self.B)
        batch_idx = tf.reshape(batch_idx, (self.B, 1, 1))
        b = tf.tile(batch_idx, [1, self.H, self.W])

        theta_slices_x = self._delta_calculator(b, px, py, theta_x)
        theta_slices_y = self._delta_calculator(b, px, py, theta_y)

        theta_slices_x = tf.cast(theta_slices_x, tf.float32)
        theta_slices_y = tf.cast(theta_slices_y, tf.float32)

        delta_x = self.Buv[:self.B] * theta_slices_x
        delta_x = tf.reduce_sum(delta_x, axis=[-2,-1])
        delta_y = self.Buv[:self.B] * theta_slices_y
        delta_y = tf.reduce_sum(delta_y, axis=[-2,-1])

        delta = tf.stack([delta_x, delta_y], axis=0)

        batch_grids = self.base_grid[:,:self.B] + delta
        return batch_grids, delta


    def _delta_calculator(
            self,
            b: tf.Tensor,
            px: tf.Tensor,
            py: tf.Tensor,
            theta: tf.Tensor
    ) -> tf.Tensor:
        """
        Calculate Bspline transformation deltas for the control grid.

        This method computes the Bspline transformation deltas for the given control grid points.

        :param b: Batch indices.
        :param px: X-coordinates of control grid points.
        :param py: Y-coordinates of control grid points.
        :param theta: Bspline transformation parameters.

        :return: Bspline transformation deltas.

        :Note: This method is used internally by the SpatialTransformerBspline layer.
        """
        px = px[0:self.B]
        py = py[0:self.B]

        t0 = self._compute_theta_slices(b, px, py, theta, 0)
        t1 = self._compute_theta_slices(b, px, py, theta, 1)
        t2 = self._compute_theta_slices(b, px, py, theta, 2)
        t3 = self._compute_theta_slices(b, px, py, theta, 3)

        t = tf.stack([t0,t1,t2,t3], axis=-1)
        return t


    def _compute_theta_slices(
            self,
            b: tf.Tensor,
            px: tf.Tensor,
            py: tf.Tensor,
            theta: tf.Tensor,
            i: int
    ) -> tf.Tensor:
        """
        Compute slices of Bspline transformation parameters for the given control grid points.

        This method extracts slices of Bspline transformation parameters corresponding
        to the specified control grid points and interpolation indices.

        :param b: Batch indices.
        :param px: X-coordinates of control grid points.
        :param py: Y-coordinates of control grid points.
        :param theta: Bspline transformation parameters.
        :param i: Interpolation index.

        :return: Slices of Bspline transformation parameters.

        :Note: This method is used internally by the SpatialTransformerBspline layer.
        """
        ti0 = tf.gather_nd(theta, tf.stack([b, py+i, px+0], 3))
        ti1 = tf.gather_nd(theta, tf.stack([b, py+i, px+1], 3))
        ti2 = tf.gather_nd(theta, tf.stack([b, py+i, px+2], 3))
        ti3 = tf.gather_nd(theta, tf.stack([b, py+i, px+3], 3))

        ti = tf.stack([ti0, ti1, ti2, ti3], axis=-1)
        return ti


    def _piece_bsplines(
            self,
            u: tf.Tensor
    ) -> tf.Tensor:
        """
        Compute the piece-wise Bspline functions for the given interpolation parameter.

        This method calculates the piece-wise Bspline functions (U0, U1, U2, U3) for the given
        interpolation parameter 'u'. These functions are used to perform interpolation during the
        spatial transformation.

        :param u: Interpolation parameter.

        :return: Piece-wise Bspline functions.

        :Note: This method is used internally by the SpatialTransformerBspline layer.
        """
        u2 = u ** 2
        u3 = u ** 3

        U0 = (-u3 + 3*u2 - 3*u + 1) / 6
        U1 = (3*u3 - 6*u2 + 4) / 6
        U2 = (-3*u3 + 3*u2 + 3*u + 1) / 6
        U3 = u3 / 6

        U = tf.stack([U0, U1, U2, U3], axis=0)
        return U


    def _bilinear_sampler(
            self,
            img: tf.Tensor,
            x: int,
            y: int
    ) -> tf.Tensor:
        """
        Perform bilinear sampling on the input feature map based on the given coordinates.

        This method implements bilinear sampling to interpolate pixel values from the input
        feature map 'img' at the specified non-integer coordinates 'x' and 'y'.

        :param img: Input feature map. Shape should be (B, H, W, C).
        :param x: X-coordinates for bilinear sampling.
        :param y: Y-coordinates for bilinear sampling.

        :return: Output feature map after bilinear sampling. Shape is (B, H, W, C).

        :Note: This method is used internally by the SpatialTransformerBspline layer.
        """
        B, H, W, C = img.shape

        # Define min and max of x and y coords
        zero = tf.zeros([], dtype=tf.int32)
        max_x = tf.cast(W-1, dtype=tf.int32)
        max_y = tf.cast(H-1, dtype=tf.int32)

        # Find corner coordinates
        x0 = tf.cast(tf.floor(x), dtype=tf.int32)
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), tf.int32)
        y1 = y0 + 1

        # Clip corner coordinates to legal values
        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)

        # Get corner pixel values
        Ia = self._pixel_intensity(img, x0, y0) # bottom left ##
        Ib = self._pixel_intensity(img, x0, y1) # top left ##
        Ic = self._pixel_intensity(img, x1, y0) # bottom right ##
        Id = self._pixel_intensity(img, x1, y1) # top right ##

        # Define weights of corner coordinates using deltas
        # First recast corner coords as float32
        x0 = tf.cast(x0, tf.float32)
        x1 = tf.cast(x1, tf.float32)
        y0 = tf.cast(y0, tf.float32)
        y1 = tf.cast(y1, tf.float32)

        # Weights
        wa = (x1 - x) * (y1 - y)
        wb = (x1 - x) * (y - y0)
        wc = (x - x0) * (y1 - y)
        wd = (x - x0) * (y - y0)

        # assert (wa + wb + wc + wd == 1.0), "Weights not equal to 1.0"

        # Add dimension for linear combination because
        # img = (B, H, W, C) and w = (B, H, W)
        wa = tf.expand_dims(wa, axis=3)
        wb = tf.expand_dims(wb, axis=3)
        wc = tf.expand_dims(wc, axis=3)
        wd = tf.expand_dims(wd, axis=3)

        # Linearly combine corner intensities with weights
        out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
        return out


    def _pixel_intensity(
            self,
            img: tf.Tensor,
            x: tf.Tensor,
            y: tf.Tensor
    ):
        """
        Gather efficiently the pixel intensities of transformed coordinates post-sampling.

        Given the input feature map 'img' and transformed coordinates 'x' and 'y', this
        method gathers pixel intensities using bilinear interpolation. The transformed
        coordinates are expecteed to be of the same shape.

        :param img: Input feature map. Shape should be (B, H, W, C).
        :param x: X-coordinates for pixel intensity gathering.
        :param y: Y-coordinates for pixel intensity gathering.

        :return: Pixel intensities at the specified coordinates. Shape is the same as `x` and `y`.

        :Note: This method is used internally by the SpatialTransformerBspline layer.
        """
        B, H, W, C = img.shape
        if B is None:
            B = 1
            x = tf.expand_dims(x[0], axis=0)
            y = tf.expand_dims(y[0], axis=0)

        batch_idx = tf.range(0, B)
        batch_idx = tf.reshape(batch_idx, (B, 1, 1))

        b = tf.tile(batch_idx, [1, H, W])
        indices = tf.stack([b, y, x], axis=3)
        return tf.gather_nd(img, indices)


    def call(
            self,
            input_fmap: tf.Tensor,
            theta: tf.Tensor | None = None,
            B: int = 1
    ) -> tf.Tensor:
        """
        Apply the Spatial Transformer B-spline layer to the input feature map.

        Given an input feature map `input_fmap` and optional transformation parameters `theta`
        and batch size `B`, this method computes the spatial transformation using B-spline
        interpolation and applies it to the input feature map.

        :param input_fmap: Input feature map to be transformed. Shape should be (B, H, W, C).
        :param theta: Transformation parameters. If None, identity transformation is used.
          Shape should be (B, 2, ny, nx), where 'ny' and 'nx' are the number of control points along
          the y and x dimensions of the B-spline grid, respectively.
        :param B: Batch size. Defaults to 1.

        :return: Transformed feature map with the same shape as `input_fmap`.

        :Note: This method is used to apply the B-spline transformation to the input feature map.
        """
        self.B = B
        out = self._transformer(input_fmap, theta)
        return out

    def __call__(
            self,
            input_fmap: tf.Tensor,
            theta: tf.Tensor | None = None,
            B: int = 1
    ) -> tf.Tensor:
        return self.call(input_fmap, theta, B)
