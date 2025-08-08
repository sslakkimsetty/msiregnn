#!/usr/bin/env python
"""
MSIregNN Visualization Examples

This script demonstrates the visualization capabilities of the MSIregNN package.
It shows various plotting functions for registration workflows, transformations,
masks, and training diagnostics.

To run this example:
    python examples/visualization_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Import visualization components
from msiregnn.visualization import (
    plot_registration_pair,
    plot_registration_workflow,
    plot_registration_metrics,
    ImagePlotter,
    TransformPlotter,
    MaskPlotter,
    DiagnosticsPlotter,
    create_publication_figure,
    plot_style
)


def generate_example_data():
    """Generate synthetic data for demonstration."""
    # Create coordinate grids
    h, w = 128, 128
    x, y = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    
    # Fixed image - circle at center
    fixed = np.exp(-((x - 0.5)**2 + (y - 0.5)**2) / 0.1)
    
    # Moving image - shifted circle
    moving = np.exp(-((x - 0.3)**2 + (y - 0.3)**2) / 0.1)
    
    # Transformed - partially aligned circle
    transformed = np.exp(-((x - 0.48)**2 + (y - 0.48)**2) / 0.1)
    
    # Vector field (simple translation)
    vector_field = np.stack([
        0.2 * np.ones((h, w)),  # x displacement
        0.2 * np.ones((h, w))   # y displacement
    ])
    
    # Attention map (gaussian centered)
    attention = np.exp(-((x - 0.5)**2 + (y - 0.5)**2) / 0.2)
    
    # Loss history (exponential decay with noise)
    iterations = np.arange(1000)
    loss_history = 0.5 * np.exp(-iterations / 200) + 0.1 + 0.05 * np.random.rand(1000)
    
    # Metrics history
    metrics = {
        'RMSE': 0.3 * np.exp(-iterations / 300) + 0.05,
        'MI': 0.5 * (1 - np.exp(-iterations / 250)) + 0.3,
        'SSIM': 0.6 * (1 - np.exp(-iterations / 200)) + 0.4
    }
    
    return {
        'fixed': fixed,
        'moving': moving,
        'transformed': transformed,
        'vector_field': vector_field,
        'attention': attention,
        'loss_history': loss_history,
        'metrics': metrics
    }


def example_basic_visualization():
    """Example 1: Basic image visualization."""
    print("Example 1: Basic Image Visualization")
    
    data = generate_example_data()
    
    # Simple registration pair visualization
    fig = plot_registration_pair(
        data['fixed'], 
        data['moving'], 
        data['transformed'],
        titles=['Fixed/Target', 'Moving/Source', 'Registered']
    )
    plt.show()


def example_image_plotter():
    """Example 2: Using the ImagePlotter class."""
    print("\nExample 2: ImagePlotter Class")
    
    data = generate_example_data()
    img_plotter = ImagePlotter()
    
    # Single image with colorbar
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    img_plotter.plot_image(
        data['fixed'],
        ax=ax,
        title='Fixed Image with Viridis Colormap',
        colormap='viridis',
        colorbar=True
    )
    plt.show()
    
    # Image comparison
    fig = img_plotter.plot_image_comparison(
        [data['fixed'], data['moving'], data['transformed']],
        titles=['Fixed', 'Moving', 'Transformed'],
        colormap='hot',
        figsize=(15, 5)
    )
    plt.show()
    
    # Grid layout for multiple channels
    msi_channels = [data['fixed'] * (i+1)/5 for i in range(6)]
    fig = img_plotter.plot_image_grid(
        msi_channels,
        titles=[f'Channel {i+1}' for i in range(6)],
        colormap='viridis',
        figsize=(12, 8)
    )
    plt.show()


def example_transformation_visualization():
    """Example 3: Transformation visualizations."""
    print("\nExample 3: Transformation Visualizations")
    
    data = generate_example_data()
    transform_plotter = TransformPlotter()
    
    # Vector field visualization
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    transform_plotter.plot_vector_field(
        data['vector_field'],
        ax=ax,
        title='Deformation Field',
        subsample=8,
        color_by_magnitude=True,
        colormap='plasma'
    )
    plt.show()
    
    # Deformation grid
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    transform_plotter.plot_deformation_grid(
        data['vector_field'],
        ax=ax,
        title='Deformation Grid',
        grid_spacing=10,
        line_color='blue',
        show_displacement=True
    )
    plt.show()
    
    # Transformation summary
    fig = transform_plotter.plot_transformation_summary(
        data['vector_field'],
        fixed_image=data['fixed'],
        moving_image=data['moving'],
        transformed_image=data['transformed']
    )
    plt.show()


def example_mask_visualization():
    """Example 4: Mask and attention visualization."""
    print("\nExample 4: Mask and Attention Visualization")
    
    data = generate_example_data()
    mask_plotter = MaskPlotter()
    
    # Attention map
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    mask_plotter.plot_attention_map(
        data['attention'],
        ax=ax,
        title='Attention Weights',
        colormap='hot',
        colorbar=True
    )
    plt.show()
    
    # Attention overlay
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    mask_plotter.plot_attention_overlay(
        data['fixed'],
        data['attention'],
        ax=ax,
        title='Attention on Fixed Image',
        attention_cmap='jet',
        attention_alpha=0.5
    )
    plt.show()
    
    # Binary mask overlay
    binary_mask = data['attention'] > 0.5
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    mask_plotter.plot_mask_overlay(
        data['fixed'],
        binary_mask,
        ax=ax,
        title='ROI Mask Overlay',
        mask_color='red',
        mask_alpha=0.3,
        contour=True
    )
    plt.show()


def example_training_diagnostics():
    """Example 5: Training diagnostics visualization."""
    print("\nExample 5: Training Diagnostics")
    
    data = generate_example_data()
    diag_plotter = DiagnosticsPlotter()
    
    # Loss curve with smoothing
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    diag_plotter.plot_loss_curve(
        data['loss_history'],
        ax=ax,
        title='Training Loss',
        log_scale=True,
        smooth=True,
        smooth_window=50,
        show_min=True
    )
    plt.show()
    
    # Metrics history
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    diag_plotter.plot_metrics_history(
        data['metrics'],
        ax=ax,
        title='Registration Metrics During Training',
        ylabel='Metric Value',
        legend_loc='right'
    )
    plt.show()
    
    # Training summary
    learning_rates = 0.001 * np.exp(-np.arange(1000) / 500)
    fig = diag_plotter.plot_training_summary(
        data['loss_history'],
        metrics=data['metrics'],
        learning_rates=learning_rates,
        figsize=(12, 12)
    )
    plt.show()
    
    # Convergence analysis
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    diag_plotter.plot_convergence_analysis(
        data['loss_history'],
        window_size=100,
        ax=ax
    )
    plt.show()


def example_composite_visualizations():
    """Example 6: Composite visualizations."""
    print("\nExample 6: Composite Visualizations")
    
    data = generate_example_data()
    
    # Complete registration workflow
    fig = plot_registration_workflow(
        data['fixed'],
        data['moving'],
        data['transformed'],
        vector_field=data['vector_field'],
        attention_map=data['attention'],
        loss_history=data['loss_history'],
        figsize=(15, 10)
    )
    plt.show()
    
    # Method comparison
    from msiregnn.visualization import plot_registration_comparison
    
    comparison_data = {
        'Affine': {
            'transformed': data['transformed'] * 0.9,
            'vector_field': data['vector_field'] * 0.5,
            'metrics': {'rmse': 0.15, 'mi': 0.75}
        },
        'B-spline': {
            'transformed': data['transformed'],
            'vector_field': data['vector_field'],
            'metrics': {'rmse': 0.10, 'mi': 0.85}
        },
        'Deep Learning': {
            'transformed': data['transformed'] * 1.05,
            'vector_field': data['vector_field'] * 1.2,
            'metrics': {'rmse': 0.08, 'mi': 0.90}
        }
    }
    
    fig = plot_registration_comparison(
        data['fixed'],
        data['moving'],
        comparison_data,
        metrics_to_show=['rmse', 'mi']
    )
    plt.show()


def example_publication_figures():
    """Example 7: Publication-ready figures."""
    print("\nExample 7: Publication-Ready Figures")
    
    data = generate_example_data()
    
    # Using style context manager
    with plot_style('publication'):
        fig = plot_registration_workflow(
            data['fixed'],
            data['moving'],
            data['transformed'],
            vector_field=data['vector_field']
        )
        # Uncomment to save:
        # fig.savefig('registration_results.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create publication figure directly
    fig = create_publication_figure(
        data={
            'fixed': data['fixed'],
            'moving': data['moving'],
            'transformed': data['transformed'],
            'vector_field': data['vector_field'],
            'attention_map': data['attention'],
            'loss_history': data['loss_history']
        },
        figure_type='registration_results',
        style='publication',
        # Uncomment to save:
        # save_path='figure_1.pdf'
    )
    plt.show()


def example_custom_styling():
    """Example 8: Custom styling and colormaps."""
    print("\nExample 8: Custom Styling")
    
    data = generate_example_data()
    
    # Access plot configuration
    from msiregnn.visualization import PlotConfig
    
    config = PlotConfig()
    print("Available colormaps:", list(config.COLORMAPS.keys()))
    
    # Use specific colormaps for different data types
    img_plotter = ImagePlotter()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    img_plotter.plot_image(
        data['fixed'],
        ax=axes[0],
        title='Anatomical Image',
        colormap=config.COLORMAPS['anatomical']
    )
    
    img_plotter.plot_image(
        data['attention'],
        ax=axes[1],
        title='Attention Map',
        colormap=config.COLORMAPS['attention']
    )
    plt.show()


def example_custom_plotter():
    """Example 9: Creating custom plotters."""
    print("\nExample 9: Custom Plotter")
    
    from msiregnn.visualization import ImagePlotter, create_figure, tensor_to_numpy
    
    class CustomRegistrationPlotter(ImagePlotter):
        """Custom plotter with additional methods."""
        
        def plot_difference_image(self, img1, img2, ax=None, **kwargs):
            """Plot the difference between two images."""
            if ax is None:
                fig, ax = create_figure(1, 1)
            
            diff = np.abs(tensor_to_numpy(img1) - tensor_to_numpy(img2))
            
            im = ax.imshow(diff, cmap='RdBu_r', vmin=-1, vmax=1)
            plt.colorbar(im, ax=ax, fraction=0.046)
            
            ax.set_title('Difference Image')
            self.setup_axes(ax, remove_ticks=True)
            
            return ax
    
    # Use custom plotter
    data = generate_example_data()
    custom_plotter = CustomRegistrationPlotter()
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    custom_plotter.plot_difference_image(
        data['fixed'],
        data['transformed'],
        ax=ax
    )
    plt.show()


def main():
    """Run all examples."""
    print("MSIregNN Visualization Examples")
    print("=" * 50)
    
    # Comment out examples you don't want to run
    example_basic_visualization()
    example_image_plotter()
    example_transformation_visualization()
    example_mask_visualization()
    example_training_diagnostics()
    example_composite_visualizations()
    example_publication_figures()
    example_custom_styling()
    example_custom_plotter()
    
    print("\nAll examples completed!")


if __name__ == "__main__":
    main()
