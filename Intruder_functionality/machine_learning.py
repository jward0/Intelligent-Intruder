"""

main python code for machine learning DNN models

- Alex Bott

"""
import tensorflow as tf
from keras.layers import Input, Dense, Flatten, TimeDistributed
from keras.models import Model
from .custom_layers import GCNLayer


class ML_Intruder:

    def __init__(self, data_shape, adjacency_shape, node_embedding_length=3, dense_units=25, learning_rate=0.001):
        self.data_shape = data_shape
        self.adjacency_shape = adjacency_shape
        self.node_embedding_length = node_embedding_length
        self.dense_units = dense_units
        self.learning_rate = learning_rate
        self.model = self.build_model()
        self.prediction = None
        self.binary_prediction = None
    
    def build_model(self):
        # Define the input shapes
        node_input = Input(shape=self.data_shape, name='node_input')
        adjacency_input = Input(shape=self.adjacency_shape, name='adjacency_input')

        # Define the model using the functional API
        gcn_output = GCNLayer(node_embedding_length=self.node_embedding_length)([node_input, adjacency_input])

        # TimeDistributed layer with Dense
        time_output = TimeDistributed(Dense(units=self.node_embedding_length, activation='relu'))(gcn_output)
        time_output = Flatten()(time_output)

        # Dense layer for the final output
        dense_output = Dense(self.dense_units, activation='sigmoid')(time_output)

        # Create the Model
        model = Model(inputs=[node_input, adjacency_input], outputs=dense_output)
        return model
    
    def compile(self):
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    def fit(self, training_data, adjacency_matrix_train, trainY, epochs, batch_size, verbose):
        self.model.fit([training_data, adjacency_matrix_train], trainY, epochs=epochs, batch_size=batch_size, verbose=verbose)
    
    def summary(self):
        self.model.summary()

    def predict(self, data, adjacency_matrix):
        self.prediction = self.model.predict([data, adjacency_matrix])
        return self.prediction

    def binary_predict(self, threshold=0.5):
        if self.prediction is None:
            raise ValueError("Must call predict() before calling binary_predict().")
        self.binary_prediction = (self.prediction > threshold).astype(int)
        return self.binary_prediction
    



