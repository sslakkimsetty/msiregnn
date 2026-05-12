import matplotlib
import pytest
import tensorflow as tf

# Use a non-interactive backend for headless CI/testing
matplotlib.use("Agg")


@pytest.fixture
def sample_image():
    """Provide a sample test image."""
    return tf.ones((1, 100, 100, 1), dtype=tf.float32)
