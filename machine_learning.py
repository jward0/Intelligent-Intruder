"""

main python code for machine learning DNN models

- Alex Bott

"""
import tensorflow as tf
import keras
from keras.layers import Input, Dense, Flatten, TimeDistributed
from keras_nlp.layers import TransformerEncoder
from keras.models import Model
import numpy as np


class ML_Intruder:

    def __init__(self, data_shape, dense_units, node_embedding_length=4, learning_rate=0.001):
        tf.random.set_seed(1234)
        self.data_shape = data_shape
        # self.adjacency_shape = adjacency_shape
        self.node_embedding_length = node_embedding_length
        self.dense_units = dense_units
        self.learning_rate = learning_rate
        self.model = self.build_model()
        self.prediction = None
        self.binary_prediction = None
    
    def build_model(self):
        # Currently gives ~80% binary accuracy after 600s
        # Define the input shapes
        node_input = Input(shape=self.data_shape, name='node_input')

        # TimeDistributed layer with Dense
        # time_output = TimeDistributed(Dense(units=self.node_embedding_length, activation='relu'))(node_input)
        time_output = TransformerEncoder(intermediate_dim=8, num_heads=3)(node_input)
        time_output = Flatten()(time_output)

        # Dense layer for the final output
        dense_output = Dense(self.dense_units, activation='sigmoid')(time_output)

        # Create the Model
        model = Model(inputs=node_input, outputs=dense_output)
        return model
    
    def compile(self):
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['binary_accuracy'])

    def train_on_batch(self, data_batch, target_batch):
        self.model.train_on_batch(data_batch, target_batch)
    
    def fit(self, training_data, train_y, epochs, batch_size, verbose):
        self.model.fit(training_data, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    
    def summary(self):
        self.model.summary()

    def predict(self, data):
        self.prediction = self.model.predict(data)
        return self.prediction

    def binary_predict(self, threshold=0.5):
        if self.prediction is None:
            raise ValueError("Must call predict() before calling binary_predict().")
        self.binary_prediction = (self.prediction > threshold).astype(int)
        return self.binary_prediction

    # @tf.function
    def evaluate_and_predict(self, train_x, train_y, window_size, f1_threshold, f2_threshold, f3_threshold, ending_timestep, vuln_data):

        # TODO: fix the stupid alternating conventions between counting forwards and backwards from window
        # TODO: starts and ends respectively, pick one and stick to it

        predictions = np.full((ending_timestep, train_x.shape[-2]), np.nan)
        replay_buffer_label = np.full((ending_timestep, train_x.shape[-2]), 0.5)
        # probabilities = np.full((ending_timestep, 1), np.nan)
        print(predictions.shape)

        time_of_attack = ending_timestep
        node_attacked = -1
        attack_outcome = -1

        rng = np.random.default_rng(1234)

        for i in range(window_size, ending_timestep):

            # Get the most recent window of data
            window_data = train_x[i-window_size:i, :, :]
            window_data = window_data[np.newaxis, ...]  # Add batch dimension

            # # Get the corresponding label
            label = train_y[i, :]
            label = label[np.newaxis, ...]  # Add batch dimension

            # Set replay buffer as all sequential data so far
            replay_buffer = train_x[:i, :, :]

            # update replay buffer labels appropriately
            vuln_threshold = vuln_data[i, :] <= 1
            replay_buffer_label[:i, vuln_threshold] = train_y[:i, vuln_threshold]

            # Sample a batch from the replay buffer
            batch_size = min(len(replay_buffer)-(window_size-1), 32)  # Example batch size=32

            # generate random batch indices
            starting_indices = rng.choice(len(replay_buffer) - (window_size-1), size=batch_size, replace=False)
            # starting_index = np.random.randint(0, len(replay_buffer) - (window_size-1), size=batch_size)

            # Generate the indices for the adjacent window
            indices = [np.arange(idx, idx + window_size) for idx in starting_indices]

            # Extract the batch samples and labels using the sampled indices
            replay_batch = replay_buffer[indices]
            replay_batch_labels = replay_buffer_label[[j+(window_size-1) for j in starting_indices]]

            # Convert the batch to arrays
            replay_batch_array = np.array(replay_batch)
            replay_batch_labels_array = np.array(replay_batch_labels)

            # TODO: custom loop this (it'll be faster)
            # Training predictions
            training_predictions = self.model(replay_batch_array, training=False).numpy()
            for ndx, j in enumerate(starting_indices):
                predictions[j+window_size-1] = training_predictions[ndx]

            # Train on the replay batch
            loss = self.model.train_on_batch(replay_batch_array, replay_batch_labels_array)
            print(f"accuracy = {loss[1]}")

            # Predict attack
            prediction = self.model(window_data, training=False).numpy()
            predictions[i] = prediction

            # Replay buffer for model predictions
            replay_buffer_predictions = predictions[:i+1, :]

            # print(prediction)
            # print(replay_buffer_predictions[-1])

            # TODO: better combination of confidence level with remaining opportunities (makes thresholding more sensible)

            # print(np.max(replay_buffer_predictions >= f1_threshold))
            # print(replay_buffer_predictions[-1])

            # probability of threshold not occurring in remaining time frame
            # p_a =
            # probability_exceed_threshold = ()
            probability_exceed_threshold = (1 - np.sum(replay_buffer_predictions.flatten() >= f1_threshold) / np.sum(replay_buffer_predictions.flatten() != np.nan))*np.square(i/ending_timestep)
            
            # prob = (1 - np.sum(replay_buffer_predictions.flatten() >= f2_threshold) / np.sum(replay_buffer_predictions.flatten() != np.nan))
            # probabilities[i] = probability_exceed_threshold

            f3 = probability_exceed_threshold*prediction.max()
            # print(i)
            # print(prediction.max())
            # print(f3)
            # max_index = prediction.argmax()
            # print(label[:,max_index])

            # Check the prediction against the threshold
            """
            if (f3 >= f3_threshold) & (prediction.max() >= f1_threshold):
                max_index = prediction.argmax()
                print(f"Threshold reached with prediction: {prediction.max()} at timestep: {i}")
                print(f"Attack successful = {label[:,max_index]==1} | at timestep: {i}")
                time_of_attack = i
                node_attacked = max_index + 1
                attack_outcome = label[:, max_index]

                break
            """
        tf.keras.backend.clear_session()
        return time_of_attack, node_attacked, attack_outcome


