import pytest
import tensorflow as tf


@pytest.fixture
def sample_image():
    """Provide a sample test image."""
    return tf.ones((1, 100, 100, 1), dtype=tf.float32)
