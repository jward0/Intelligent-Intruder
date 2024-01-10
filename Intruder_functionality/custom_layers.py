"""

custom Tensorflow layers

- Alex Bott

"""


import tensorflow as tf
from keras import layers, initializers, activations

class GCNLayer(layers.Layer):
    def __init__(self, node_embedding_length):
        super(GCNLayer, self).__init__()

        # Self weights
        self.a_0 = self.add_weight(
            shape=(1, 1),
            initializer=initializers.HeUniform(),
            trainable=True,
            dtype='float32'
        )
        # 1-hop weights
        self.a_1 = self.add_weight(
            shape=(1, 1),
            initializer=initializers.HeUniform(),
            trainable=True,
            dtype='float32'
        )

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'a_0': self.a_0,
            'a_1': self.a_1,
        })
        return config

    def call(self, inputs):

        node_values = inputs[0]
        adjacency_matrix = inputs[1]
        summed_out = tf.math.add(self.a_0*node_values,
                                 tf.linalg.matmul(adjacency_matrix,
                                                  self.a_1*node_values)
                                 )

        return activations.relu(summed_out, alpha=0.3)
    pass