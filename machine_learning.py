"""

main python code for machine learning DNN models

- Alex Bott

"""
import tensorflow as tf
from keras.layers import Input, Dense, Flatten, TimeDistributed
from keras.models import Model
import numpy as np


class ML_Intruder:

    def __init__(self, data_shape, dense_units, node_embedding_length=4, learning_rate=0.001):
        self.data_shape = data_shape
        # self.adjacency_shape = adjacency_shape
        self.node_embedding_length = node_embedding_length
        self.dense_units = dense_units
        self.learning_rate = learning_rate
        self.model = self.build_model()
        self.prediction = None
        self.binary_prediction = None
    
    def build_model(self):
        # Define the input shapes
        node_input = Input(shape=self.data_shape, name='node_input')

        # TimeDistributed layer with Dense
        time_output = TimeDistributed(Dense(units=self.node_embedding_length, activation='relu'))(node_input)
        time_output = Flatten()(time_output)

        # Dense layer for the final output
        dense_output = Dense(self.dense_units, activation='sigmoid')(time_output)

        # Create the Model
        model = Model(inputs=node_input, outputs=dense_output)
        return model
    
    def compile(self):
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    def train_on_batch(self, data_batch, target_batch):
        self.model.train_on_batch(data_batch, target_batch)
    
    def fit(self, training_data, trainY, epochs, batch_size, verbose):
        self.model.fit(training_data, trainY, epochs=epochs, batch_size=batch_size, verbose=verbose)
    
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
    
    def evaluate_and_predict(self, trainX, trainY, window_size, f1_threshold, f2_threshold, f3_threshold, ending_timestep, vuln_data):

        predictions = np.full((ending_timestep, trainX.shape[-2]), np.nan)
        replay_buffer_label = np.full((ending_timestep, trainX.shape[-2]), 0.5)
        # probabilities = np.full((ending_timestep, 1), np.nan)
        print(predictions.shape)

        time_of_attack = ending_timestep
        node_attacked = -1
        attack_outcome = -1

        for i in range(window_size, ending_timestep):

            # Get the most recent window of data
            window_data = trainX[i-window_size:i,:,:]
            window_data = window_data[np.newaxis, ...]  # Add batch dimension

            # # Get the corresponding label
            label = trainY[i,:]
            label = label[np.newaxis, ...]  # Add batch dimension

            # Set replay buffer as all sequential data so far
            replay_buffer = trainX[0:i,:,:]

            #update replay buffer labels appropriatly
            vuln_threshold = vuln_data[i,:] <= 1
            replay_buffer_label[0:i, vuln_threshold] = trainY[0:i, vuln_threshold]


            # Sample a batch from the replay buffer
            batch_size = min(len(replay_buffer), 32)  # Example batch size=32

            # generate random batch indices
            starting_index = np.random.randint(0, len(replay_buffer) - (window_size-1),size=batch_size)

            # Generate the indices for the adjacent window
            indices = [np.arange(idx, idx + window_size) for idx in starting_index]

            # Extract the batch samples and labels using the sampled indices
            replay_batch = replay_buffer[indices]
            replay_batch_labels = replay_buffer_label[[i+(window_size-1) for i in starting_index]]

            # Convert the batch to arrays
            replay_batch_array = np.array(replay_batch)
            replay_batch_labels_array = np.array(replay_batch_labels)

            # Train on the replay batch
            self.train_on_batch(replay_batch_array, replay_batch_labels_array)

            # Predict attack
            prediction = self.predict(window_data)

            # Replay buffer for model predictions
            predictions[[i+(window_size-1) for i in starting_index]] = prediction
            replay_buffer_predictions = predictions[0:i,:]

            # probability of threshold not occuring in remaining time frame
            probability_exceed_threshold = (1 - np.sum(replay_buffer_predictions.flatten() >= f2_threshold) / np.sum(replay_buffer_predictions.flatten() != np.nan))*np.square(i/ending_timestep)
            
            # prob = (1 - np.sum(replay_buffer_predictions.flatten() >= f2_threshold) / np.sum(replay_buffer_predictions.flatten() != np.nan))
            # probabilities[i] = probability_exceed_threshold

            f3 = probability_exceed_threshold*prediction.max()
            # print(prediction.max())
            # print(f3)
            # max_index = prediction.argmax()
            # print(label[:,max_index])


            # Check the prediction against the threshold
            if (f3 >= f3_threshold)&(prediction.max()>=f1_threshold):
                max_index = prediction.argmax()
                print(f"Threshold reached with prediction: {prediction.max()} at timestep: {i}")
                print(f"Attack succesful = {label[:,max_index]==1} | at timestep: {i}")
                time_of_attack = i
                node_attacked = max_index + 1
                attack_outcome = label[:,max_index]

                break
        
        return time_of_attack, node_attacked, attack_outcome


