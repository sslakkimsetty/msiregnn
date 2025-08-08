"""MSIregNN visualization module.

This module provides comprehensive plotting functionality for visualizing
registration workflows, transformations, masks, and training diagnostics.
"""

# Core utilities
# Composite visualizations
from .composite import (
    create_publication_figure,
    plot_ablation_study,
    plot_multi_scale_registration,
    plot_registration_comparison,
    plot_registration_workflow,
)
from .core import (
    BasePlotter,
    PlotConfig,
    create_figure,
    normalize_image,
    plot_style,
    save_figure,
    tensor_to_numpy,
)

# Diagnostics visualization
from .diagnostics import DiagnosticsPlotter, plot_registration_metrics

# Image visualization
from .images import ImagePlotter, plot_msi_channels, plot_registration_pair

# Mask and attention visualization
from .masks import MaskPlotter, plot_attention_analysis

# Transformation visualization
from .transforms import TransformPlotter, plot_bspline_control_points

__all__ = [
    # Core
    'PlotConfig',
    'plot_style',
    'tensor_to_numpy',
    'normalize_image',
    'create_figure',
    'save_figure',
    'BasePlotter',

    # Images
    'ImagePlotter',
    'plot_registration_pair',
    'plot_msi_channels',

    # Transforms
    'TransformPlotter',
    'plot_bspline_control_points',

    # Masks
    'MaskPlotter',
    'plot_attention_analysis',

    # Diagnostics
    'DiagnosticsPlotter',
    'plot_registration_metrics',

    # Composite
    'plot_registration_workflow',
    'plot_multi_scale_registration',
    'plot_registration_comparison',
    'plot_ablation_study',
    'create_publication_figure'
]
