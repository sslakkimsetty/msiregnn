"""Core plotting utilities and base classes for MSIregNN visualization."""

from contextlib import contextmanager
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.axes import Axes
from matplotlib.figure import Figure


class PlotConfig:
    """Configuration for plot styling and defaults."""

    # Default figure settings
    DEFAULT_FIGSIZE = (8, 6)
    DEFAULT_DPI = 100
    DEFAULT_FONT_SIZE = 12

    # Color schemes
    COLORMAPS = {
        'anatomical': 'gray',
        'msi': 'viridis',
        'overlay': 'hot',
        'mask': 'binary_r',
        'attention': 'plasma',
        'vector_field': 'twilight',
        'error': 'RdBu_r'
    }

    # Style presets
    STYLES = {
        'publication': {
            'font.size': 10,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.dpi': 300
        },
        'presentation': {
            'font.size': 14,
            'axes.labelsize': 16,
            'axes.titlesize': 18,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'legend.fontsize': 14,
            'figure.dpi': 150
        },
        'notebook': {
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.dpi': 100
        }
    }


@contextmanager
def plot_style(style: str = 'notebook'):
    """Context manager for temporary plot styling.
    
    Args:
        style: Style preset name ('publication', 'presentation', 'notebook')
        
    Example:
        with plot_style('publication'):
            plot_registration_results(model)
    """
    if style not in PlotConfig.STYLES:
        raise ValueError(f"Unknown style: {style}. Choose from {list(PlotConfig.STYLES.keys())}")

    old_params = plt.rcParams.copy()
    try:
        plt.rcParams.update(PlotConfig.STYLES[style])
        yield
    finally:
        plt.rcParams.update(old_params)


def tensor_to_numpy(tensor: Union[tf.Tensor, np.ndarray]) -> np.ndarray:
    """Convert tensor to numpy array safely."""
    if isinstance(tensor, tf.Tensor):
        return tensor.numpy()
    return np.asarray(tensor)


def normalize_image(
    image: Union[tf.Tensor, np.ndarray],
    percentile: Optional[Tuple[float, float]] = None
) -> np.ndarray:
    """Normalize image for display.
    
    Args:
        image: Input image
        percentile: Optional percentile range for robust normalization
        
    Returns:
        Normalized image in range [0, 1]
    """
    img = tensor_to_numpy(image).squeeze()

    if percentile is not None:
        vmin, vmax = np.percentile(img, percentile)
    else:
        vmin, vmax = img.min(), img.max()

    if vmax > vmin:
        img = (img - vmin) / (vmax - vmin)
    else:
        img = np.zeros_like(img)

    return np.clip(img, 0, 1)


def create_figure(
    nrows: int = 1,
    ncols: int = 1,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: Optional[int] = None,
    tight_layout: bool = True,
    **kwargs
) -> Tuple[Figure, Union[Axes, np.ndarray]]:
    """Create figure with consistent styling.
    
    Args:
        nrows: Number of subplot rows
        ncols: Number of subplot columns
        figsize: Figure size (width, height) in inches
        dpi: Dots per inch
        tight_layout: Whether to use tight layout
        **kwargs: Additional arguments for plt.subplots
        
    Returns:
        Tuple of (figure, axes)
    """
    if figsize is None:
        figsize = (PlotConfig.DEFAULT_FIGSIZE[0] * ncols / 1.5,
                  PlotConfig.DEFAULT_FIGSIZE[1] * nrows / 1.5)

    if dpi is None:
        dpi = PlotConfig.DEFAULT_DPI

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi, **kwargs)

    if tight_layout:
        fig.tight_layout()

    return fig, axes


def save_figure(
    fig: Figure,
    filename: str,
    dpi: Optional[int] = None,
    bbox_inches: str = 'tight',
    transparent: bool = False,
    **kwargs
) -> None:
    """Save figure with consistent settings.
    
    Args:
        fig: Figure to save
        filename: Output filename
        dpi: Resolution (defaults to figure dpi)
        bbox_inches: Bounding box setting
        transparent: Whether to save with transparent background
        **kwargs: Additional arguments for savefig
    """
    if dpi is None:
        dpi = fig.dpi

    fig.savefig(
        filename,
        dpi=dpi,
        bbox_inches=bbox_inches,
        transparent=transparent,
        **kwargs
    )


class BasePlotter:
    """Base class for specialized plotters."""

    def __init__(self, style: str = 'notebook'):
        self.style = style
        self.config = PlotConfig()

    def setup_axes(
        self,
        ax: Axes,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        remove_ticks: bool = False,
        grid: bool = False
    ) -> None:
        """Configure axes with common settings."""
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)

        if remove_ticks:
            ax.set_xticks([])
            ax.set_yticks([])

        ax.grid(grid)

    @contextmanager
    def plotting_context(self):
        """Context manager for plotting with configured style."""
        with plot_style(self.style):
            yield
