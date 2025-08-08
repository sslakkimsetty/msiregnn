"""Training diagnostics and metrics visualization for MSIregNN."""

from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .core import BasePlotter, create_figure, tensor_to_numpy


class DiagnosticsPlotter(BasePlotter):
    """Specialized plotter for training diagnostics and metrics."""

    def plot_loss_curve(
        self,
        losses: Union[List[float], np.ndarray],
        ax: Optional[Axes] = None,
        title: str = 'Training Loss',
        xlabel: str = 'Iteration',
        ylabel: str = 'Loss',
        log_scale: bool = False,
        smooth: bool = False,
        smooth_window: int = 10,
        show_min: bool = True,
        **kwargs
    ) -> Axes:
        """Plot training loss curve with optional smoothing.
        
        Args:
            losses: List or array of loss values
            ax: Matplotlib axes
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            log_scale: Whether to use log scale for y-axis
            smooth: Whether to show smoothed curve
            smooth_window: Window size for smoothing
            show_min: Whether to mark minimum loss
            **kwargs: Additional arguments for plot
            
        Returns:
            The axes object
        """
        if ax is None:
            fig, ax = create_figure(1, 1)

        losses = np.array(losses)
        iterations = np.arange(len(losses))

        # Plot raw losses
        ax.plot(iterations, losses, alpha=0.3 if smooth else 1.0,
               label='Raw' if smooth else None, **kwargs)

        # Add smoothed curve if requested
        if smooth and len(losses) > smooth_window:
            smoothed = np.convolve(losses,
                                 np.ones(smooth_window)/smooth_window,
                                 mode='valid')
            smooth_iters = iterations[smooth_window//2:-(smooth_window//2-1)]
            ax.plot(smooth_iters, smoothed, label='Smoothed', **kwargs)

        # Mark minimum
        if show_min:
            min_idx = np.argmin(losses)
            ax.plot(min_idx, losses[min_idx], 'ro', markersize=8)
            ax.annotate(f'Min: {losses[min_idx]:.4f}',
                       xy=(min_idx, losses[min_idx]),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

        # Formatting
        if log_scale:
            ax.set_yscale('log')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        if smooth:
            ax.legend()

        return ax

    def plot_learning_rate_schedule(
        self,
        learning_rates: Union[List[float], np.ndarray],
        ax: Optional[Axes] = None,
        title: str = 'Learning Rate Schedule',
        xlabel: str = 'Iteration',
        ylabel: str = 'Learning Rate',
        log_scale: bool = True,
        **kwargs
    ) -> Axes:
        """Plot learning rate schedule.
        
        Args:
            learning_rates: Learning rate values over iterations
            ax: Matplotlib axes
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            log_scale: Whether to use log scale for y-axis
            **kwargs: Additional arguments
            
        Returns:
            The axes object
        """
        if ax is None:
            fig, ax = create_figure(1, 1)

        iterations = np.arange(len(learning_rates))
        ax.plot(iterations, learning_rates, **kwargs)

        if log_scale:
            ax.set_yscale('log')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        return ax

    def plot_metrics_history(
        self,
        metrics: Dict[str, Union[List[float], np.ndarray]],
        ax: Optional[Axes] = None,
        title: str = 'Metrics History',
        xlabel: str = 'Iteration',
        ylabel: str = 'Metric Value',
        legend_loc: str = 'best',
        **kwargs
    ) -> Axes:
        """Plot multiple metrics over training iterations.
        
        Args:
            metrics: Dictionary of metric names to values
            ax: Matplotlib axes
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            legend_loc: Legend location
            **kwargs: Additional arguments
            
        Returns:
            The axes object
        """
        if ax is None:
            fig, ax = create_figure(1, 1)

        for name, values in metrics.items():
            iterations = np.arange(len(values))
            ax.plot(iterations, values, label=name, **kwargs)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc=legend_loc)
        ax.grid(True, alpha=0.3)

        return ax

    def plot_parameter_distribution(
        self,
        parameters: Dict[str, Union[tf.Tensor, np.ndarray]],
        ax: Optional[Axes] = None,
        title: str = 'Parameter Distributions',
        bins: int = 50,
        **kwargs
    ) -> Axes:
        """Plot histograms of model parameters.
        
        Args:
            parameters: Dictionary of parameter names to tensors
            ax: Matplotlib axes
            title: Plot title
            bins: Number of histogram bins
            **kwargs: Additional arguments
            
        Returns:
            The axes object
        """
        if ax is None:
            fig, ax = create_figure(1, 1)

        for name, param in parameters.items():
            values = tensor_to_numpy(param).flatten()
            ax.hist(values, bins=bins, alpha=0.5, label=name, **kwargs)

        ax.set_xlabel('Parameter Value')
        ax.set_ylabel('Count')
        ax.set_title(title)
        ax.legend()

        return ax

    def plot_gradient_flow(
        self,
        named_parameters: List[Tuple[str, Union[tf.Tensor, np.ndarray]]],
        ax: Optional[Axes] = None,
        title: str = 'Gradient Flow',
        **kwargs
    ) -> Axes:
        """Plot gradient flow through network layers.
        
        Args:
            named_parameters: List of (name, gradient) tuples
            ax: Matplotlib axes
            title: Plot title
            **kwargs: Additional arguments
            
        Returns:
            The axes object
        """
        if ax is None:
            fig, ax = create_figure(1, 1, figsize=(12, 6))

        # Extract gradient statistics
        names = []
        mean_grads = []
        max_grads = []

        for name, grad in named_parameters:
            if grad is not None:
                grad_np = tensor_to_numpy(grad)
                names.append(name)
                mean_grads.append(np.abs(grad_np).mean())
                max_grads.append(np.abs(grad_np).max())

        # Plot
        x = np.arange(len(names))
        ax.bar(x - 0.2, mean_grads, 0.4, label='Mean', **kwargs)
        ax.bar(x + 0.2, max_grads, 0.4, label='Max', **kwargs)

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel('Gradient Magnitude')
        ax.set_title(title)
        ax.legend()
        ax.set_yscale('log')

        return ax

    def plot_training_summary(
        self,
        losses: Union[List[float], np.ndarray],
        metrics: Optional[Dict[str, Union[List[float], np.ndarray]]] = None,
        learning_rates: Optional[Union[List[float], np.ndarray]] = None,
        figsize: Optional[Tuple[float, float]] = None,
        **kwargs
    ) -> Figure:
        """Create a comprehensive training summary plot.
        
        Args:
            losses: Loss values
            metrics: Optional dictionary of metrics
            learning_rates: Optional learning rate values
            figsize: Figure size
            **kwargs: Additional arguments
            
        Returns:
            The figure object
        """
        # Determine layout
        n_plots = 1  # Loss is always plotted
        if metrics:
            n_plots += 1
        if learning_rates is not None:
            n_plots += 1

        if figsize is None:
            figsize = (12, 4 * n_plots)

        fig, axes = create_figure(n_plots, 1, figsize=figsize)
        if n_plots == 1:
            axes = [axes]

        # Plot loss
        self.plot_loss_curve(losses, ax=axes[0], smooth=True, **kwargs)

        # Plot metrics if provided
        plot_idx = 1
        if metrics:
            self.plot_metrics_history(metrics, ax=axes[plot_idx], **kwargs)
            plot_idx += 1

        # Plot learning rate if provided
        if learning_rates is not None:
            self.plot_learning_rate_schedule(learning_rates, ax=axes[plot_idx], **kwargs)

        plt.tight_layout()
        return fig

    def plot_convergence_analysis(
        self,
        losses: Union[List[float], np.ndarray],
        window_size: int = 50,
        ax: Optional[Axes] = None,
        title: str = 'Convergence Analysis',
        **kwargs
    ) -> Axes:
        """Analyze and plot convergence behavior.
        
        Args:
            losses: Loss values
            window_size: Window for computing statistics
            ax: Matplotlib axes
            title: Plot title
            **kwargs: Additional arguments
            
        Returns:
            The axes object
        """
        if ax is None:
            fig, ax = create_figure(1, 1, figsize=(10, 6))

        losses = np.array(losses)

        # Compute rolling statistics
        if len(losses) > window_size:
            rolling_mean = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
            rolling_std = np.array([np.std(losses[i:i+window_size])
                                  for i in range(len(losses) - window_size)])

            x = np.arange(window_size//2, len(losses) - window_size//2)

            # Plot mean with confidence interval
            ax.plot(x, rolling_mean, label='Rolling Mean', **kwargs)
            ax.fill_between(x,
                          rolling_mean - 2*rolling_std,
                          rolling_mean + 2*rolling_std,
                          alpha=0.3, label='95% CI')

            # Add convergence rate
            if len(rolling_mean) > 10:
                # Fit exponential decay to later part
                x_fit = np.arange(len(rolling_mean) // 2, len(rolling_mean))
                y_fit = rolling_mean[len(rolling_mean) // 2:]

                # Log transform for linear fit
                log_y = np.log(y_fit + 1e-10)
                coeffs = np.polyfit(x_fit, log_y, 1)
                convergence_rate = -coeffs[0]

                ax.text(0.02, 0.98, f'Convergence rate: {convergence_rate:.4f}',
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.plot(losses, **kwargs)

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax


def plot_registration_metrics(
    metrics_dict: Dict[str, Union[List[float], np.ndarray]],
    figsize: Optional[Tuple[float, float]] = None,
    **kwargs
) -> Figure:
    """Convenience function to plot common registration metrics.
    
    Args:
        metrics_dict: Dictionary with keys like 'rmse', 'mi', 'ssim'
        figsize: Figure size
        **kwargs: Additional arguments
        
    Returns:
        The figure object
    """
    plotter = DiagnosticsPlotter()

    # Separate metrics by type
    similarity_metrics = {}
    error_metrics = {}

    for name, values in metrics_dict.items():
        if name.lower() in ['mi', 'nmi', 'ssim', 'ncc']:
            similarity_metrics[name] = values
        else:
            error_metrics[name] = values

    n_plots = sum([len(similarity_metrics) > 0, len(error_metrics) > 0])

    if figsize is None:
        figsize = (12, 4 * n_plots)

    fig, axes = create_figure(n_plots, 1, figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    plot_idx = 0

    # Plot similarity metrics (higher is better)
    if similarity_metrics:
        plotter.plot_metrics_history(
            similarity_metrics,
            ax=axes[plot_idx],
            title='Similarity Metrics (Higher is Better)',
            ylabel='Metric Value',
            **kwargs
        )
        plot_idx += 1

    # Plot error metrics (lower is better)
    if error_metrics:
        plotter.plot_metrics_history(
            error_metrics,
            ax=axes[plot_idx],
            title='Error Metrics (Lower is Better)',
            ylabel='Error Value',
            **kwargs
        )

    plt.tight_layout()
    return fig
