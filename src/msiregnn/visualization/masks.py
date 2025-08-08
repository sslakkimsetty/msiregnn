"""Mask and attention visualization for MSIregNN."""

from typing import List, Optional, Tuple, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from .core import BasePlotter, create_figure, normalize_image, tensor_to_numpy
from .images import ImagePlotter


class MaskPlotter(BasePlotter):
    """Specialized plotter for masks and attention maps."""

    def plot_binary_mask(
        self,
        mask: Union[tf.Tensor, np.ndarray],
        ax: Optional[Axes] = None,
        title: Optional[str] = None,
        true_color: str = 'white',
        false_color: str = 'black',
        alpha: float = 1.0,
        show_stats: bool = True,
        **kwargs
    ) -> Axes:
        """Plot a binary mask with optional statistics.
        
        Args:
            mask: Binary mask tensor
            ax: Matplotlib axes
            title: Plot title
            true_color: Color for True/1 values
            false_color: Color for False/0 values
            alpha: Transparency
            show_stats: Whether to show mask statistics
            **kwargs: Additional arguments for imshow
            
        Returns:
            The axes object
        """
        if ax is None:
            fig, ax = create_figure(1, 1)

        # Convert mask
        mask_array = tensor_to_numpy(mask).squeeze()

        # Create colormap
        colors = [false_color, true_color]
        n_bins = 2
        cmap = mcolors.LinearSegmentedColormap.from_list('binary_mask', colors, n_bins)

        # Plot mask
        im = ax.imshow(mask_array, cmap=cmap, vmin=0, vmax=1,
                      alpha=alpha, **kwargs)

        # Add statistics if requested
        if show_stats:
            coverage = np.mean(mask_array) * 100
            stats_text = f'Coverage: {coverage:.1f}%'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round',
                   facecolor='white', alpha=0.8))

        # Setup axes
        ax.set_xticks([])
        ax.set_yticks([])

        if title:
            ax.set_title(title)

        return ax

    def plot_attention_map(
        self,
        attention: Union[tf.Tensor, np.ndarray],
        ax: Optional[Axes] = None,
        title: Optional[str] = None,
        colormap: str = 'hot',
        colorbar: bool = True,
        threshold: Optional[float] = None,
        normalize: bool = True,
        **kwargs
    ) -> Axes:
        """Plot an attention map with customizable visualization.
        
        Args:
            attention: Attention weights tensor
            ax: Matplotlib axes
            title: Plot title
            colormap: Colormap for attention values
            colorbar: Whether to show colorbar
            threshold: Optional threshold for binary display
            normalize: Whether to normalize to [0, 1]
            **kwargs: Additional arguments for imshow
            
        Returns:
            The axes object
        """
        if ax is None:
            fig, ax = create_figure(1, 1)

        # Process attention map
        att_map = tensor_to_numpy(attention).squeeze()

        if normalize and att_map.max() > att_map.min():
            att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min())

        if threshold is not None:
            att_map = (att_map > threshold).astype(float)

        # Plot attention
        im = ax.imshow(att_map, cmap=colormap, vmin=0, vmax=1, **kwargs)

        if colorbar:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Attention Weight', rotation=270, labelpad=15)

        # Setup axes
        ax.set_xticks([])
        ax.set_yticks([])

        if title:
            ax.set_title(title)

        return ax

    def plot_mask_overlay(
        self,
        image: Union[tf.Tensor, np.ndarray],
        mask: Union[tf.Tensor, np.ndarray],
        ax: Optional[Axes] = None,
        title: Optional[str] = None,
        mask_color: str = 'red',
        mask_alpha: float = 0.3,
        contour: bool = True,
        contour_color: str = 'red',
        contour_width: float = 2.0,
        **kwargs
    ) -> Axes:
        """Overlay a mask on an image with optional contour.
        
        Args:
            image: Background image
            mask: Binary mask to overlay
            ax: Matplotlib axes
            title: Plot title
            mask_color: Color for mask overlay
            mask_alpha: Transparency of mask
            contour: Whether to show mask contour
            contour_color: Color of contour
            contour_width: Width of contour line
            **kwargs: Additional arguments
            
        Returns:
            The axes object
        """
        if ax is None:
            fig, ax = create_figure(1, 1)

        # Plot base image
        img_array = normalize_image(image)
        ax.imshow(img_array, cmap='gray', **kwargs)

        # Process mask
        mask_array = tensor_to_numpy(mask).squeeze() > 0.5

        # Create colored mask
        mask_rgba = np.zeros((*mask_array.shape, 4))
        mask_rgba[..., :3] = mcolors.to_rgb(mask_color)
        mask_rgba[..., 3] = mask_array * mask_alpha

        # Overlay mask
        ax.imshow(mask_rgba)

        # Add contour if requested
        if contour:
            from skimage import measure

            try:
                contours = measure.find_contours(mask_array, 0.5)
                for contour_coords in contours:
                    ax.plot(contour_coords[:, 1], contour_coords[:, 0],
                           color=contour_color, linewidth=contour_width)
            except ImportError:
                # Fallback if skimage not available
                pass

        # Setup axes
        ax.set_xticks([])
        ax.set_yticks([])

        if title:
            ax.set_title(title)

        return ax

    def plot_attention_overlay(
        self,
        image: Union[tf.Tensor, np.ndarray],
        attention: Union[tf.Tensor, np.ndarray],
        ax: Optional[Axes] = None,
        title: Optional[str] = None,
        attention_cmap: str = 'jet',
        attention_alpha: float = 0.5,
        threshold: Optional[float] = None,
        **kwargs
    ) -> Axes:
        """Overlay attention weights on an image.
        
        Args:
            image: Background image
            attention: Attention weights
            ax: Matplotlib axes
            title: Plot title
            attention_cmap: Colormap for attention
            attention_alpha: Transparency of attention overlay
            threshold: Optional threshold for attention display
            **kwargs: Additional arguments
            
        Returns:
            The axes object
        """
        if ax is None:
            fig, ax = create_figure(1, 1)

        # Plot base image
        img_array = normalize_image(image)
        ax.imshow(img_array, cmap='gray', **kwargs)

        # Process attention
        att_array = normalize_image(attention)

        if threshold is not None:
            att_array = np.where(att_array > threshold, att_array, 0)

        # Overlay attention
        im = ax.imshow(att_array, cmap=attention_cmap, alpha=attention_alpha,
                      vmin=0, vmax=1, **kwargs)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Attention', rotation=270, labelpad=15)

        # Setup axes
        ax.set_xticks([])
        ax.set_yticks([])

        if title:
            ax.set_title(title)

        return ax

    def plot_roi_boxes(
        self,
        image: Union[tf.Tensor, np.ndarray],
        boxes: List[Tuple[int, int, int, int]],
        ax: Optional[Axes] = None,
        title: Optional[str] = None,
        box_color: str = 'red',
        box_width: float = 2.0,
        labels: Optional[List[str]] = None,
        **kwargs
    ) -> Axes:
        """Plot ROI boxes on an image.
        
        Args:
            image: Background image
            boxes: List of boxes as (x, y, width, height)
            ax: Matplotlib axes
            title: Plot title
            box_color: Color of boxes
            box_width: Width of box edges
            labels: Optional labels for each box
            **kwargs: Additional arguments
            
        Returns:
            The axes object
        """
        if ax is None:
            fig, ax = create_figure(1, 1)

        # Plot image
        img_plotter = ImagePlotter()
        img_plotter.plot_image(image, ax=ax)

        # Add boxes
        for i, (x, y, w, h) in enumerate(boxes):
            rect = Rectangle((x, y), w, h, linewidth=box_width,
                           edgecolor=box_color, facecolor='none')
            ax.add_patch(rect)

            # Add label if provided
            if labels and i < len(labels):
                ax.text(x, y - 5, labels[i], color=box_color,
                       fontsize=10, weight='bold')

        if title:
            ax.set_title(title)

        return ax

    def plot_multi_channel_masks(
        self,
        masks: List[Union[tf.Tensor, np.ndarray]],
        labels: Optional[List[str]] = None,
        colors: Optional[List[str]] = None,
        ax: Optional[Axes] = None,
        title: Optional[str] = None,
        alpha: float = 0.5,
        **kwargs
    ) -> Axes:
        """Plot multiple masks with different colors.
        
        Args:
            masks: List of binary masks
            labels: Optional labels for each mask
            colors: Colors for each mask (auto-generated if None)
            ax: Matplotlib axes
            title: Plot title
            alpha: Transparency for overlays
            **kwargs: Additional arguments
            
        Returns:
            The axes object
        """
        if ax is None:
            fig, ax = create_figure(1, 1)

        # Default colors
        if colors is None:
            colors = plt.cm.tab10(np.linspace(0, 1, len(masks)))[:, :3]

        # Create composite image
        composite = np.zeros((*masks[0].shape[:2], 3))

        for i, mask in enumerate(masks):
            mask_array = tensor_to_numpy(mask).squeeze() > 0.5
            color = mcolors.to_rgb(colors[i]) if isinstance(colors[i], str) else colors[i]

            for c in range(3):
                composite[..., c] += mask_array * color[c] * alpha

        # Clip values
        composite = np.clip(composite, 0, 1)

        # Plot composite
        ax.imshow(composite, **kwargs)

        # Add legend if labels provided
        if labels:
            from matplotlib.patches import Patch
            patches = [Patch(color=colors[i], label=labels[i])
                      for i in range(len(labels))]
            ax.legend(handles=patches, loc='upper right')

        # Setup axes
        ax.set_xticks([])
        ax.set_yticks([])

        if title:
            ax.set_title(title)

        return ax


def plot_attention_analysis(
    attention_maps: List[Union[tf.Tensor, np.ndarray]],
    layer_names: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    colormap: str = 'hot',
    **kwargs
) -> Figure:
    """Plot attention maps from multiple layers.
    
    Args:
        attention_maps: List of attention maps from different layers
        layer_names: Names of the layers
        figsize: Figure size
        colormap: Colormap for attention
        **kwargs: Additional arguments
        
    Returns:
        The figure object
    """
    plotter = MaskPlotter()

    n_maps = len(attention_maps)
    nrows = int(np.ceil(np.sqrt(n_maps)))
    ncols = int(np.ceil(n_maps / nrows))

    if figsize is None:
        figsize = (4 * ncols, 4 * nrows)

    fig, axes = create_figure(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if nrows * ncols > 1 else [axes]

    if layer_names is None:
        layer_names = [f'Layer {i+1}' for i in range(n_maps)]

    for i, (att_map, name) in enumerate(zip(attention_maps, layer_names)):
        if i < len(axes):
            plotter.plot_attention_map(att_map, ax=axes[i], title=name,
                                     colormap=colormap, **kwargs)

    # Hide unused subplots
    for i in range(n_maps, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    return fig
