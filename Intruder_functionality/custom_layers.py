"""

custom Tensorflow layers

- Alex Bott

"""


import tensorflow as tf
from keras import layers, initializers, activations

# class GCNLayer(layers.Layer):
#     def __init__(self, node_embedding_length):
#         super(GCNLayer, self).__init__()

#         # Self weights
#         self.a_0 = self.add_weight(
#             shape=(1, 1),
#             initializer=initializers.HeUniform(),
#             trainable=True,
#             dtype='float32'
#         )
#         # 1-hop weights
#         self.a_1 = self.add_weight(
#             shape=(1, 1),
#             initializer=initializers.HeUniform(),
#             trainable=True,
#             dtype='float32'
#         )

#     def get_config(self):
#         config = super().get_config().copy()
#         config.update({
#             'a_0': self.a_0,
#             'a_1': self.a_1,
#         })
#         return config

#     def call(self, inputs):


#         node_values = inputs[0]
#         adjacency_matrix = inputs[1]

#         print("Node values (input):", node_values)
#         print("Adjacency matrix (input):", adjacency_matrix)

#         weighted_nodes_self = self.a_0 * node_values
#         weighted_nodes_neigh = tf.linalg.matmul(adjacency_matrix, self.a_1 * node_values)

#         print("Weighted node values by self weights (a_0):", weighted_nodes_self)
#         print("Weighted node values by neighbor weights (a_1):", weighted_nodes_neigh)

#         summed_out = tf.math.add(weighted_nodes_self, weighted_nodes_neigh)
        
#         print("Summed output:", summed_out)

#         relu_out = activations.relu(summed_out, alpha=0.3)

#         print("Output after ReLU activation:", relu_out)

#         return relu_out
#     pass

class GCNLayer(layers.Layer):
    def __init__(self, node_embedding_length):
        super(GCNLayer, self).__init__()
        self.node_embedding_length = node_embedding_length
        
        # Initialize weights for transforming node features
        self.kernel_self = self.add_weight(
            shape=(self.node_embedding_length, self.node_embedding_length),
            initializer=initializers.GlorotUniform(),
            trainable=True,
            name='kernel_self'
        )
        
        # Initialize weights for transforming aggregated neighbor features
        self.kernel_neighbors = self.add_weight(
            shape=(self.node_embedding_length, self.node_embedding_length),
            initializer=initializers.GlorotUniform(),
            trainable=True,
            name='kernel_neighbors'
        )

    def call(self, inputs):
        node_values = inputs[0]  # Shape: (num_nodes, feature_dim)
        adjacency_matrix = inputs[1]  # Shape: (num_nodes, num_nodes)

        # Transform node features with self-weights
        node_features_transformed = tf.matmul(node_values, self.kernel_self)

        # Aggregate neighbor features and transform
        aggregated_neighbors = tf.matmul(adjacency_matrix, node_values)
        neighbors_features_transformed = tf.matmul(aggregated_neighbors, self.kernel_neighbors)

        # Combine self and neighbor feature transformations
        output = node_features_transformed + neighbors_features_transformed

        # Apply a non-linear activation function
        return activations.relu(output)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'node_embedding_length': self.node_embedding_length,
            'kernel_self': self.kernel_self,
            'kernel_neighbors': self.kernel_neighbors
        })
        return config
    