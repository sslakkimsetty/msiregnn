import tensorflow as tf
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D


class SpatialAttentionMap(tf.keras.layers.Layer):
    def __init__(self):
        super(SpatialAttentionMap, self).__init__()
        self.conv1 = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(filters=1, kernel_size=3, padding='same', activation='sigmoid')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = Activation('relu')(x)
        attention = self.conv2(x)
        return attention

class AttentionModel(tf.keras.models.Model):
    def __init__(self):
        super(AttentionModel, self).__init__()
        self.spatial_attention_map = SpatialAttentionMap()
        self.conv = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')

    def call(self, inputs):
        attention_map = self.spatial_attention_map(inputs)
        x = self.conv(inputs)
        x = tf.multiply(x, attention_map)
        return x

    def create_attention_masks(self, inputs):
        attention_map = self.spatial_attention_map(inputs)
        return attention_map
