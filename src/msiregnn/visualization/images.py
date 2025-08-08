"""Image visualization functions for MSIregNN."""

from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .core import BasePlotter, create_figure, normalize_image, tensor_to_numpy


class ImagePlotter(BasePlotter):
    """Specialized plotter for image visualization."""

    def plot_image(
        self,
        image: Union[tf.Tensor, np.ndarray],
        ax: Optional[Axes] = None,
        title: Optional[str] = None,
        colormap: Optional[str] = None,
        colorbar: bool = False,
        percentile: Optional[Tuple[float, float]] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        **kwargs
    ) -> Axes:
        """Plot a single image with customizable options.
        
        Args:
            image: Input image (2D or 3D tensor/array)
            ax: Matplotlib axes (creates new if None)
            title: Title for the plot
            colormap: Colormap name (defaults based on image type)
            colorbar: Whether to add a colorbar
            percentile: Percentile range for normalization
            vmin, vmax: Manual intensity range
            **kwargs: Additional arguments for imshow
            
        Returns:
            The axes object
        """
        if ax is None:
            fig, ax = create_figure(1, 1)

        # Process image
        img = tensor_to_numpy(image).squeeze()

        # Handle RGB images
        if img.ndim == 3 and img.shape[-1] in [3, 4]:
            if percentile is not None or vmin is not None or vmax is not None:
                # Normalize each channel
                for i in range(img.shape[-1]):
                    img[..., i] = normalize_image(img[..., i], percentile)
        else:
            # Single channel image
            if percentile is not None:
                img = normalize_image(img, percentile)
            elif vmin is None and vmax is None:
                vmin, vmax = img.min(), img.max()

        # Default colormap selection
        if colormap is None:
            if img.ndim == 3:
                colormap = None  # No colormap for RGB
            else:
                colormap = self.config.COLORMAPS['anatomical']

        # Plot image
        im = ax.imshow(img, cmap=colormap, vmin=vmin, vmax=vmax, **kwargs)

        # Add colorbar if requested
        if colorbar and img.ndim == 2:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Setup axes
        self.setup_axes(ax, title=title, remove_ticks=True)

        return ax

    def plot_image_comparison(
        self,
        images: List[Union[tf.Tensor, np.ndarray]],
        titles: Optional[List[str]] = None,
        colormap: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
        colorbar: bool = False,
        percentile: Optional[Tuple[float, float]] = None,
        **kwargs
    ) -> Figure:
        """Plot multiple images side by side for comparison.
        
        Args:
            images: List of images to compare
            titles: Optional titles for each image
            colormap: Colormap to use for all images
            figsize: Figure size
            colorbar: Whether to add colorbars
            percentile: Percentile range for normalization
            **kwargs: Additional arguments for imshow
            
        Returns:
            The figure object
        """
        n_images = len(images)
        fig, axes = create_figure(1, n_images, figsize=figsize)

        if n_images == 1:
            axes = [axes]

        if titles is None:
            titles = [None] * n_images

        for ax, img, title in zip(axes, images, titles):
            self.plot_image(
                img, ax=ax, title=title, colormap=colormap,
                colorbar=colorbar, percentile=percentile, **kwargs
            )

        return fig

    def plot_image_grid(
        self,
        images: List[Union[tf.Tensor, np.ndarray]],
        nrows: Optional[int] = None,
        ncols: Optional[int] = None,
        titles: Optional[List[str]] = None,
        colormap: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
        **kwargs
    ) -> Figure:
        """Plot images in a grid layout.
        
        Args:
            images: List of images
            nrows, ncols: Grid dimensions (auto-calculated if not provided)
            titles: Optional titles for each image
            colormap: Colormap to use
            figsize: Figure size
            **kwargs: Additional arguments for plot_image
            
        Returns:
            The figure object
        """
        n_images = len(images)

        # Auto-calculate grid dimensions
        if nrows is None and ncols is None:
            ncols = int(np.ceil(np.sqrt(n_images)))
            nrows = int(np.ceil(n_images / ncols))
        elif nrows is None:
            nrows = int(np.ceil(n_images / ncols))
        elif ncols is None:
            ncols = int(np.ceil(n_images / nrows))

        fig, axes = create_figure(nrows, ncols, figsize=figsize)
        axes = axes.flatten() if nrows * ncols > 1 else [axes]

        if titles is None:
            titles = [None] * n_images

        for i, (img, title) in enumerate(zip(images, titles)):
            if i < len(axes):
                self.plot_image(img, ax=axes[i], title=title,
                              colormap=colormap, **kwargs)

        # Hide empty subplots
        for i in range(n_images, len(axes)):
            axes[i].axis('off')

        return fig

    def plot_image_overlay(
        self,
        background: Union[tf.Tensor, np.ndarray],
        overlay: Union[tf.Tensor, np.ndarray],
        ax: Optional[Axes] = None,
        alpha: float = 0.5,
        background_cmap: str = 'gray',
        overlay_cmap: str = 'hot',
        title: Optional[str] = None,
        **kwargs
    ) -> Axes:
        """Plot one image overlaid on another.
        
        Args:
            background: Background image
            overlay: Overlay image
            ax: Matplotlib axes
            alpha: Transparency of overlay
            background_cmap: Colormap for background
            overlay_cmap: Colormap for overlay
            title: Plot title
            **kwargs: Additional arguments for imshow
            
        Returns:
            The axes object
        """
        if ax is None:
            fig, ax = create_figure(1, 1)

        # Plot background
        bg = normalize_image(background)
        ax.imshow(bg, cmap=background_cmap, **kwargs)

        # Plot overlay
        ov = normalize_image(overlay)
        ax.imshow(ov, cmap=overlay_cmap, alpha=alpha, **kwargs)

        self.setup_axes(ax, title=title, remove_ticks=True)

        return ax


def plot_registration_pair(
    fixed: Union[tf.Tensor, np.ndarray],
    moving: Union[tf.Tensor, np.ndarray],
    transformed: Optional[Union[tf.Tensor, np.ndarray]] = None,
    titles: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    **kwargs
) -> Figure:
    """Convenience function to plot registration image pair.
    
    Args:
        fixed: Fixed/target image
        moving: Moving/source image
        transformed: Optional transformed moving image
        titles: Titles for subplots
        figsize: Figure size
        **kwargs: Additional arguments for plotting
        
    Returns:
        The figure object
    """
    plotter = ImagePlotter()

    images = [fixed, moving]
    if titles is None:
        titles = ['Fixed Image', 'Moving Image']

    if transformed is not None:
        images.append(transformed)
        titles.append('Transformed Moving')

    return plotter.plot_image_comparison(images, titles=titles,
                                        figsize=figsize, **kwargs)


def plot_msi_channels(
    msi_data: Union[tf.Tensor, np.ndarray],
    channels: Optional[List[int]] = None,
    titles: Optional[List[str]] = None,
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    colormap: str = 'viridis',
    **kwargs
) -> Figure:
    """Plot multiple MSI channels in a grid.
    
    Args:
        msi_data: MSI data tensor (channels x height x width)
        channels: List of channel indices to plot (all if None)
        titles: Optional titles for each channel
        nrows, ncols: Grid dimensions
        colormap: Colormap for MSI data
        **kwargs: Additional plotting arguments
        
    Returns:
        The figure object
    """
    plotter = ImagePlotter()

    data = tensor_to_numpy(msi_data)
    if channels is None:
        channels = list(range(data.shape[0]))

    images = [data[ch] for ch in channels]

    if titles is None:
        titles = [f'Channel {ch}' for ch in channels]

    return plotter.plot_image_grid(images, titles=titles, nrows=nrows,
                                  ncols=ncols, colormap=colormap, **kwargs)
