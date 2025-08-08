"""Transformation and vector field visualization for MSIregNN."""

from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .core import BasePlotter, create_figure, tensor_to_numpy


class TransformPlotter(BasePlotter):
    """Specialized plotter for transformation visualizations."""

    def plot_vector_field(
        self,
        vector_field: Union[tf.Tensor, np.ndarray],
        ax: Optional[Axes] = None,
        title: Optional[str] = None,
        subsample: int = 1,
        scale: Optional[float] = None,
        color_by_magnitude: bool = True,
        colormap: str = 'viridis',
        arrow_width: float = 0.002,
        show_grid: bool = False,
        **kwargs
    ) -> Axes:
        """Plot a dense vector field with customizable appearance.
        
        Args:
            vector_field: Vector field tensor (2, H, W) or (H, W, 2)
            ax: Matplotlib axes
            title: Plot title
            subsample: Subsampling factor for vector display
            scale: Arrow scaling factor
            color_by_magnitude: Color arrows by displacement magnitude
            colormap: Colormap for magnitude coloring
            arrow_width: Width of arrows
            show_grid: Whether to show grid lines
            **kwargs: Additional arguments for quiver
            
        Returns:
            The axes object
        """
        if ax is None:
            fig, ax = create_figure(1, 1)

        # Convert and reshape vector field
        vf = tensor_to_numpy(vector_field)
        if vf.shape[0] == 2 and len(vf.shape) == 3:
            # Shape is (2, H, W)
            vy, vx = vf[0], vf[1]
        elif vf.shape[-1] == 2 and len(vf.shape) == 3:
            # Shape is (H, W, 2)
            vy, vx = vf[..., 0], vf[..., 1]
        else:
            raise ValueError(f"Invalid vector field shape: {vf.shape}")

        # Create coordinate grids
        h, w = vy.shape
        y, x = np.mgrid[0:h:subsample, 0:w:subsample]
        vy_sub = vy[::subsample, ::subsample]
        vx_sub = vx[::subsample, ::subsample]

        # Calculate magnitude for coloring
        if color_by_magnitude:
            magnitude = np.sqrt(vx_sub**2 + vy_sub**2)
            colors = magnitude.flatten()
        else:
            colors = None

        # Auto-scale if not provided
        if scale is None:
            max_displacement = np.max(np.sqrt(vx_sub**2 + vy_sub**2))
            if max_displacement > 0:
                scale = 1.0 / max_displacement * min(h, w) / 20
            else:
                scale = 1.0

        # Plot vector field
        q = ax.quiver(
            x, y, vx_sub, -vy_sub,  # Negative vy for correct orientation
            colors,
            cmap=colormap,
            scale=scale,
            scale_units='xy',
            angles='xy',
            width=arrow_width,
            **kwargs
        )

        # Add colorbar if coloring by magnitude
        if color_by_magnitude and colors is not None:
            cbar = plt.colorbar(q, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Displacement Magnitude', rotation=270, labelpad=15)

        # Setup axes
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)  # Invert y-axis for image coordinates
        ax.set_aspect('equal')

        if not show_grid:
            ax.set_xticks([])
            ax.set_yticks([])

        if title:
            ax.set_title(title)

        return ax

    def plot_deformation_grid(
        self,
        vector_field: Union[tf.Tensor, np.ndarray],
        ax: Optional[Axes] = None,
        title: Optional[str] = None,
        grid_spacing: int = 10,
        line_color: str = 'blue',
        line_width: float = 1.0,
        show_displacement: bool = True,
        **kwargs
    ) -> Axes:
        """Plot a deformation grid showing the transformation.
        
        Args:
            vector_field: Vector field tensor (2, H, W) or (H, W, 2)
            ax: Matplotlib axes
            title: Plot title
            grid_spacing: Spacing between grid lines
            line_color: Color of grid lines
            line_width: Width of grid lines
            show_displacement: Whether to show the displacement field
            **kwargs: Additional arguments
            
        Returns:
            The axes object
        """
        if ax is None:
            fig, ax = create_figure(1, 1)

        # Convert vector field
        vf = tensor_to_numpy(vector_field)
        if vf.shape[0] == 2 and len(vf.shape) == 3:
            vy, vx = vf[0], vf[1]
        elif vf.shape[-1] == 2 and len(vf.shape) == 3:
            vy, vx = vf[..., 0], vf[..., 1]
        else:
            raise ValueError(f"Invalid vector field shape: {vf.shape}")

        h, w = vy.shape

        # Create grid points
        y_lines = np.arange(0, h, grid_spacing)
        x_lines = np.arange(0, w, grid_spacing)

        # Plot horizontal lines
        for y in y_lines:
            x_coords = np.arange(0, w)
            y_coords = np.full_like(x_coords, y, dtype=float)

            if show_displacement and y < h:
                x_coords = x_coords + vx[y, :]
                y_coords = y_coords + vy[y, :]

            ax.plot(x_coords, y_coords, color=line_color,
                   linewidth=line_width, **kwargs)

        # Plot vertical lines
        for x in x_lines:
            y_coords = np.arange(0, h)
            x_coords = np.full_like(y_coords, x, dtype=float)

            if show_displacement and x < w:
                x_coords = x_coords + vx[:, x]
                y_coords = y_coords + vy[:, x]

            ax.plot(x_coords, y_coords, color=line_color,
                   linewidth=line_width, **kwargs)

        # Setup axes
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

        if title:
            ax.set_title(title)

        return ax

    def plot_jacobian_determinant(
        self,
        vector_field: Union[tf.Tensor, np.ndarray],
        ax: Optional[Axes] = None,
        title: Optional[str] = None,
        colormap: str = 'RdBu_r',
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        colorbar: bool = True,
        **kwargs
    ) -> Axes:
        """Plot the Jacobian determinant of a transformation.
        
        Args:
            vector_field: Vector field tensor
            ax: Matplotlib axes
            title: Plot title
            colormap: Colormap (RdBu_r shows expansion/compression well)
            vmin, vmax: Color scale limits
            colorbar: Whether to show colorbar
            **kwargs: Additional arguments for imshow
            
        Returns:
            The axes object
        """
        if ax is None:
            fig, ax = create_figure(1, 1)

        # Calculate Jacobian determinant
        jac_det = self._compute_jacobian_determinant(vector_field)

        # Default color limits centered at 1 (no volume change)
        if vmin is None:
            vmin = max(0.5, jac_det.min())
        if vmax is None:
            vmax = min(2.0, jac_det.max())

        # Plot
        im = ax.imshow(jac_det, cmap=colormap, vmin=vmin, vmax=vmax, **kwargs)

        if colorbar:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Jacobian Determinant', rotation=270, labelpad=15)

        # Setup axes
        ax.set_xticks([])
        ax.set_yticks([])

        if title:
            ax.set_title(title)

        return ax

    def plot_transformation_summary(
        self,
        vector_field: Union[tf.Tensor, np.ndarray],
        fixed_image: Optional[Union[tf.Tensor, np.ndarray]] = None,
        moving_image: Optional[Union[tf.Tensor, np.ndarray]] = None,
        transformed_image: Optional[Union[tf.Tensor, np.ndarray]] = None,
        figsize: Optional[Tuple[float, float]] = None,
        **kwargs
    ) -> Figure:
        """Create a comprehensive transformation visualization.
        
        Args:
            vector_field: The transformation vector field
            fixed_image: Optional fixed image
            moving_image: Optional moving image
            transformed_image: Optional transformed moving image
            figsize: Figure size
            **kwargs: Additional arguments
            
        Returns:
            The figure object
        """
        # Determine layout based on available images
        n_image_plots = sum(x is not None for x in
                          [fixed_image, moving_image, transformed_image])

        if n_image_plots > 0:
            nrows = 2
            ncols = max(2, n_image_plots)
        else:
            nrows = 1
            ncols = 3

        if figsize is None:
            figsize = (5 * ncols, 5 * nrows)

        fig, axes = create_figure(nrows, ncols, figsize=figsize)
        if nrows == 1:
            axes = axes.reshape(1, -1)

        # First row: images
        col = 0
        if fixed_image is not None:
            from .images import ImagePlotter
            img_plotter = ImagePlotter()
            img_plotter.plot_image(fixed_image, ax=axes[0, col],
                                 title='Fixed Image')
            col += 1

        if moving_image is not None:
            from .images import ImagePlotter
            img_plotter = ImagePlotter()
            img_plotter.plot_image(moving_image, ax=axes[0, col],
                                 title='Moving Image')
            col += 1

        if transformed_image is not None:
            from .images import ImagePlotter
            img_plotter = ImagePlotter()
            img_plotter.plot_image(transformed_image, ax=axes[0, col],
                                 title='Transformed Moving')
            col += 1

        # Hide unused image plots
        for i in range(col, ncols):
            if n_image_plots > 0:
                axes[0, i].axis('off')

        # Second row (or first if no images): transformation visualizations
        row = 1 if n_image_plots > 0 else 0

        # Vector field
        self.plot_vector_field(vector_field, ax=axes[row, 0],
                             title='Vector Field', subsample=10)

        # Deformation grid
        self.plot_deformation_grid(vector_field, ax=axes[row, 1],
                                 title='Deformation Grid')

        # Jacobian determinant
        if ncols > 2:
            self.plot_jacobian_determinant(vector_field, ax=axes[row, 2],
                                         title='Jacobian Determinant')

        # Hide any remaining unused plots
        for i in range(3, ncols):
            axes[row, i].axis('off')

        plt.tight_layout()
        return fig

    def _compute_jacobian_determinant(
        self,
        vector_field: Union[tf.Tensor, np.ndarray]
    ) -> np.ndarray:
        """Compute Jacobian determinant of transformation.
        
        Args:
            vector_field: Vector field (2, H, W) or (H, W, 2)
            
        Returns:
            Jacobian determinant array
        """
        vf = tensor_to_numpy(vector_field)

        # Reshape to (H, W, 2) if needed
        if vf.shape[0] == 2 and len(vf.shape) == 3:
            vf = np.transpose(vf, (1, 2, 0))

        # Compute gradients
        dy_dx = np.gradient(vf[..., 0], axis=1)
        dy_dy = np.gradient(vf[..., 0], axis=0)
        dx_dx = np.gradient(vf[..., 1], axis=1)
        dx_dy = np.gradient(vf[..., 1], axis=0)

        # Jacobian matrix elements (including identity)
        j11 = 1 + dy_dy
        j12 = dy_dx
        j21 = dx_dy
        j22 = 1 + dx_dx

        # Determinant
        det = j11 * j22 - j12 * j21

        return det


def plot_bspline_control_points(
    control_points: Union[tf.Tensor, np.ndarray],
    image_shape: Tuple[int, int],
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    point_size: float = 50,
    point_color: str = 'red',
    show_grid: bool = True,
    **kwargs
) -> Axes:
    """Plot B-spline control points.
    
    Args:
        control_points: Control point positions
        image_shape: Shape of the image domain
        ax: Matplotlib axes
        title: Plot title
        point_size: Size of control points
        point_color: Color of control points
        show_grid: Whether to show connecting grid
        **kwargs: Additional arguments
        
    Returns:
        The axes object
    """
    plotter = TransformPlotter()

    if ax is None:
        fig, ax = create_figure(1, 1)

    # Convert control points
    cp = tensor_to_numpy(control_points)

    # Plot control points
    # Implementation depends on control point format
    # This is a placeholder that should be adapted to actual format

    ax.set_xlim(0, image_shape[1])
    ax.set_ylim(image_shape[0], 0)
    ax.set_aspect('equal')

    if title:
        ax.set_title(title)

    return ax
