"""Composite and workflow visualizations for MSIregNN."""

from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.figure import Figure

from .core import create_figure
from .diagnostics import DiagnosticsPlotter
from .images import ImagePlotter
from .masks import MaskPlotter
from .transforms import TransformPlotter


def plot_registration_workflow(
    fixed: Union[tf.Tensor, np.ndarray],
    moving: Union[tf.Tensor, np.ndarray],
    transformed: Union[tf.Tensor, np.ndarray],
    vector_field: Optional[Union[tf.Tensor, np.ndarray]] = None,
    attention_map: Optional[Union[tf.Tensor, np.ndarray]] = None,
    loss_history: Optional[List[float]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    style: str = 'notebook',
    **kwargs
) -> Figure:
    """Create a comprehensive registration workflow visualization.
    
    Args:
        fixed: Fixed/target image
        moving: Moving/source image  
        transformed: Transformed moving image
        vector_field: Optional transformation vector field
        attention_map: Optional attention weights
        loss_history: Optional loss values during training
        figsize: Figure size
        style: Plot style preset
        **kwargs: Additional arguments
        
    Returns:
        The figure object
    """
    # Determine layout based on available data
    has_vector_field = vector_field is not None
    has_attention = attention_map is not None
    has_loss = loss_history is not None

    # Calculate grid dimensions
    n_image_rows = 1
    n_extra_rows = sum([has_vector_field, has_attention, has_loss])
    total_rows = n_image_rows + (1 if n_extra_rows > 0 else 0)

    if figsize is None:
        figsize = (15, 5 * total_rows)

    fig = plt.figure(figsize=figsize)

    # Create grid spec
    if n_extra_rows > 0:
        gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1])
    else:
        gs = gridspec.GridSpec(1, 3)

    # Plot images in first row
    img_plotter = ImagePlotter(style=style)

    ax1 = fig.add_subplot(gs[0, 0])
    img_plotter.plot_image(fixed, ax=ax1, title='Fixed Image')

    ax2 = fig.add_subplot(gs[0, 1])
    img_plotter.plot_image(moving, ax=ax2, title='Moving Image')

    ax3 = fig.add_subplot(gs[0, 2])
    img_plotter.plot_image(transformed, ax=ax3, title='Transformed Moving')

    # Plot additional visualizations in second row
    if n_extra_rows > 0:
        col = 0

        if has_vector_field:
            ax = fig.add_subplot(gs[1, col])
            transform_plotter = TransformPlotter(style=style)
            transform_plotter.plot_vector_field(
                vector_field, ax=ax, title='Deformation Field',
                subsample=10, **kwargs
            )
            col += 1

        if has_attention:
            ax = fig.add_subplot(gs[1, col])
            mask_plotter = MaskPlotter(style=style)
            mask_plotter.plot_attention_overlay(
                fixed, attention_map, ax=ax,
                title='Attention on Fixed', **kwargs
            )
            col += 1

        if has_loss:
            ax = fig.add_subplot(gs[1, col])
            diag_plotter = DiagnosticsPlotter(style=style)
            diag_plotter.plot_loss_curve(
                loss_history, ax=ax, smooth=True,
                title='Training Loss', **kwargs
            )

    plt.tight_layout()
    return fig


def plot_multi_scale_registration(
    images_by_scale: Dict[str, Dict[str, Union[tf.Tensor, np.ndarray]]],
    figsize: Optional[Tuple[float, float]] = None,
    **kwargs
) -> Figure:
    """Visualize multi-scale registration results.
    
    Args:
        images_by_scale: Dictionary with scale names as keys, each containing
                        'fixed', 'moving', 'transformed' images
        figsize: Figure size
        **kwargs: Additional arguments
        
    Returns:
        The figure object
    """
    scales = list(images_by_scale.keys())
    n_scales = len(scales)

    if figsize is None:
        figsize = (12, 4 * n_scales)

    fig, axes = create_figure(n_scales, 3, figsize=figsize)
    if n_scales == 1:
        axes = axes.reshape(1, -1)

    img_plotter = ImagePlotter()

    for i, scale in enumerate(scales):
        scale_data = images_by_scale[scale]

        # Fixed image
        img_plotter.plot_image(
            scale_data['fixed'], ax=axes[i, 0],
            title=f'Fixed - {scale}'
        )

        # Moving image
        img_plotter.plot_image(
            scale_data['moving'], ax=axes[i, 1],
            title=f'Moving - {scale}'
        )

        # Transformed image
        img_plotter.plot_image(
            scale_data['transformed'], ax=axes[i, 2],
            title=f'Transformed - {scale}'
        )

    plt.tight_layout()
    return fig


def plot_registration_comparison(
    fixed: Union[tf.Tensor, np.ndarray],
    moving: Union[tf.Tensor, np.ndarray],
    results: Dict[str, Dict[str, Any]],
    metrics_to_show: List[str] = ['rmse', 'mi'],
    figsize: Optional[Tuple[float, float]] = None,
    **kwargs
) -> Figure:
    """Compare multiple registration methods.
    
    Args:
        fixed: Fixed image
        moving: Moving image
        results: Dictionary with method names as keys, each containing:
                - 'transformed': transformed image
                - 'vector_field': optional vector field
                - 'metrics': dictionary of metric values
        metrics_to_show: Which metrics to display
        figsize: Figure size
        **kwargs: Additional arguments
        
    Returns:
        The figure object
    """
    methods = list(results.keys())
    n_methods = len(methods)

    # Layout: methods in rows, columns for images and metrics
    n_cols = 4  # fixed, moving, transformed, vector field

    if figsize is None:
        figsize = (4 * n_cols, 4 * (n_methods + 1))

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(n_methods + 1, n_cols, height_ratios=[1] * n_methods + [0.5])

    img_plotter = ImagePlotter()
    transform_plotter = TransformPlotter()

    # Plot reference images in first row
    ax = fig.add_subplot(gs[0, 0])
    img_plotter.plot_image(fixed, ax=ax, title='Fixed Image')

    ax = fig.add_subplot(gs[0, 1])
    img_plotter.plot_image(moving, ax=ax, title='Moving Image')

    # Hide unused reference slots
    for col in range(2, n_cols):
        ax = fig.add_subplot(gs[0, col])
        ax.axis('off')

    # Plot each method's results
    for i, method in enumerate(methods):
        row = i + 1
        method_data = results[method]

        # Skip fixed image column
        ax = fig.add_subplot(gs[row, 0])
        ax.axis('off')
        ax.text(0.5, 0.5, method, transform=ax.transAxes,
               ha='center', va='center', fontsize=14, weight='bold')

        # Skip moving image column
        ax = fig.add_subplot(gs[row, 1])
        ax.axis('off')

        # Transformed image
        ax = fig.add_subplot(gs[row, 2])
        img_plotter.plot_image(
            method_data['transformed'], ax=ax,
            title='Transformed'
        )

        # Vector field if available
        if 'vector_field' in method_data and method_data['vector_field'] is not None:
            ax = fig.add_subplot(gs[row, 3])
            transform_plotter.plot_vector_field(
                method_data['vector_field'], ax=ax,
                title='Deformation', subsample=10
            )
        else:
            ax = fig.add_subplot(gs[row, 3])
            ax.axis('off')

    # Add metrics comparison in bottom row
    if 'metrics' in results[methods[0]]:
        ax = fig.add_subplot(gs[-1, :])

        # Prepare data for bar plot
        metric_values = {metric: [] for metric in metrics_to_show}
        for method in methods:
            for metric in metrics_to_show:
                if metric in results[method]['metrics']:
                    metric_values[metric].append(results[method]['metrics'][metric])
                else:
                    metric_values[metric].append(0)

        # Create grouped bar plot
        x = np.arange(len(methods))
        width = 0.8 / len(metrics_to_show)

        for i, metric in enumerate(metrics_to_show):
            offset = (i - len(metrics_to_show)/2 + 0.5) * width
            ax.bar(x + offset, metric_values[metric], width, label=metric.upper())

        ax.set_xlabel('Method')
        ax.set_ylabel('Metric Value')
        ax.set_title('Metrics Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    return fig


def plot_ablation_study(
    baseline_results: Dict[str, Any],
    ablation_results: Dict[str, Dict[str, Any]],
    metric: str = 'rmse',
    figsize: Optional[Tuple[float, float]] = None,
    **kwargs
) -> Figure:
    """Visualize ablation study results.
    
    Args:
        baseline_results: Results from baseline model
        ablation_results: Dictionary with component names as keys
        metric: Which metric to compare
        figsize: Figure size
        **kwargs: Additional arguments
        
    Returns:
        The figure object
    """
    components = list(ablation_results.keys())

    if figsize is None:
        figsize = (10, 6)

    fig, (ax1, ax2) = create_figure(1, 2, figsize=figsize)

    # Bar plot of metric values
    baseline_value = baseline_results['metrics'][metric]
    values = [baseline_value] + [ablation_results[c]['metrics'][metric]
                                for c in components]
    labels = ['Baseline'] + components

    bars = ax1.bar(labels, values)
    bars[0].set_color('green')

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}', ha='center', va='bottom')

    ax1.set_ylabel(metric.upper())
    ax1.set_title(f'Ablation Study - {metric.upper()}')
    ax1.tick_params(axis='x', rotation=45)

    # Relative change plot
    relative_changes = [(baseline_value - v) / baseline_value * 100
                       for v in values[1:]]

    colors = ['red' if rc < 0 else 'green' for rc in relative_changes]
    ax2.bar(components, relative_changes, color=colors)

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_ylabel('Relative Change (%)')
    ax2.set_title('Impact of Removing Components')
    ax2.tick_params(axis='x', rotation=45)

    # Add value labels
    for i, (comp, rc) in enumerate(zip(components, relative_changes)):
        ax2.text(i, rc + (1 if rc > 0 else -1), f'{rc:.1f}%',
                ha='center', va='bottom' if rc > 0 else 'top')

    plt.tight_layout()
    return fig


def create_publication_figure(
    data: Dict[str, Any],
    figure_type: str = 'registration_results',
    style: str = 'publication',
    save_path: Optional[str] = None,
    dpi: int = 300,
    **kwargs
) -> Figure:
    """Create publication-ready figures with consistent styling.
    
    Args:
        data: Data dictionary specific to figure type
        figure_type: Type of figure to create
        style: Style preset
        save_path: Optional path to save figure
        dpi: Resolution for saving
        **kwargs: Additional arguments
        
    Returns:
        The figure object
    """
    from .core import plot_style, save_figure

    with plot_style(style):
        if figure_type == 'registration_results':
            fig = plot_registration_workflow(**data, style=style, **kwargs)
        elif figure_type == 'comparison':
            fig = plot_registration_comparison(**data, **kwargs)
        elif figure_type == 'ablation':
            fig = plot_ablation_study(**data, **kwargs)
        elif figure_type == 'multi_scale':
            fig = plot_multi_scale_registration(**data, **kwargs)
        else:
            raise ValueError(f"Unknown figure type: {figure_type}")

    if save_path:
        save_figure(fig, save_path, dpi=dpi)

    return fig
