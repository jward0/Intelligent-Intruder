"""

main python code for machine learning DNN models

- Alex Bott

"""
import tensorflow as tf
import keras
from keras.layers import Input, Dense, Flatten, TimeDistributed, BatchNormalization, Reshape, Conv1D, LSTM, SimpleRNN, Bidirectional, GaussianNoise, Dropout, LayerNormalization
# from keras_nlp.layers import TransformerEncoder, TransformerDecoder
from keras.models import Model
from keras.losses import binary_crossentropy
import numpy as np

import matplotlib.pyplot as plt

# @tf.function
# def custom_loss(y_true, y_pred):
#
#     positive_weight = 5.0
#     largest_weight = 5.0
#
#     bce_loss = binary_crossentropy(tf.expand_dims(y_true, -1), tf.expand_dims(y_pred, -1))
#     max_pred_indices = tf.argmax(y_pred, axis=-1)
#     mask = tf.one_hot(max_pred_indices, depth=y_pred.shape[-1])
#     weighted_loss = bce_loss + (largest_weight - 1) * bce_loss * mask + (positive_weight - 1) * bce_loss * y_true
#
#     return tf.reduce_mean(weighted_loss)
#
#
# def custom_loss_test(y_true, y_pred):
#
#     positive_weight = 5.0
#     largest_weight = 5.0
#
#     bce_loss = binary_crossentropy(tf.expand_dims(y_true, -1), tf.expand_dims(y_pred, -1))
#     max_pred_indices = tf.argmax(y_pred, axis=-1)
#     mask = tf.one_hot(max_pred_indices, depth=y_pred.shape[-1])
#     weighted_loss = bce_loss + (largest_weight - 1) * bce_loss * mask + (positive_weight - 1) * bce_loss * y_true
#
#     return bce_loss, weighted_loss


class ML_Intruder:

    def __init__(self, data_shape, dense_units, node_embedding_length=4, learning_rate=0.001, l1_magnitude=1.0, batch_size=1, pos_weight=1.0):
        super().__init__()
        np.random.seed(1234)
        tf.random.set_seed(1234)
        self.data_shape = data_shape
        # self.adjacency_shape = adjacency_shape
        self.node_embedding_length = node_embedding_length
        self.dense_units = dense_units
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.pos_weight = pos_weight
        self.lar_weight = 1.0
        self.l1_magnitude = l1_magnitude
        self.model = self.build_model()
        self.prediction = None
        self.binary_prediction = None
        self.bce = keras.losses.BinaryCrossentropy(reduction="sum_over_batch_size")
        self.precision = keras.metrics.Precision(top_k=1)
        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.positives = 0
        self.negatives = 0

    # @tf.function
    def custom_loss(self, y_true, y_pred):

        positive_weight = self.pos_weight * ((self.negatives / np.max((self.positives, 1))) - 1)
        largest_weight = self.lar_weight

        bce_loss = binary_crossentropy(tf.expand_dims(y_true, -1), tf.expand_dims(y_pred, -1))
        max_pred_indices = tf.argmax(y_pred, axis=-1)
        mask = tf.one_hot(max_pred_indices, depth=y_pred.shape[-1])
        weighted_loss = bce_loss + (largest_weight - 1) * bce_loss * mask + positive_weight * bce_loss * y_true

        return tf.reduce_mean(weighted_loss)

    def build_model(self):
        # Currently gives ~80% binary accuracy after 600s
        # Define the input shapes
        """
        node_input = Input(shape=self.data_shape, name='node_input')
        # reshaped_input = Reshape((self.data_shape[0], self.data_shape[1]*self.data_shape[2]))(node_input)
        normalised_input = BatchNormalization(axis=-2)(node_input)
        # Node-wise convolutional layer
        node_feature_extractor = Conv1D(filters=1, kernel_size=1, activation='leaky_relu')
        # TD layer
        node_features = TimeDistributed(node_feature_extractor)(normalised_input)
        # out = TransformerEncoder(intermediate_dim=8, num_heads=1, activation='leaky_relu')(node_features)
        # out = LSTM(units=self.data_shape[1])(Reshape((self.data_shape[0], self.data_shape[1]))(node_features))
        out = SimpleRNN(self.data_shape[1], activation='sigmoid', kernel_regularizer='l1')(Reshape((self.data_shape[0], self.data_shape[1]))(node_features))
        # node_features = Flatten()(node_features)
        # out = Dense(self.data_shape[1], activation='sigmoid')(node_features)
        model = Model(inputs=node_input, outputs=out)
        print(model.summary())
        """

        # The following gives ~30% test precision
        """
        node_input = Input(shape=self.data_shape, name='node_input')
        transformer_output = Flatten()(TransformerEncoder(intermediate_dim=8,
                                                          num_heads=3,
                                                          activation='leaky_relu')(node_input))
        dense_output = Dense(self.data_shape[1],
                             activation='sigmoid')(transformer_output)
        model = Model(inputs=node_input, outputs=dense_output)
        print(model.summary())
        """
        node_input = Input(shape=self.data_shape, name='node_input')

        node_feature_extractor_1 = Conv1D(filters=6, kernel_size=1, activation='leaky_relu')
        node_feature_extractor_2 = Conv1D(filters=1, kernel_size=1, activation='leaky_relu')

        td_1 = TimeDistributed(node_feature_extractor_1)(node_input)
        td_2 = TimeDistributed(node_feature_extractor_2)(td_1)

        transformer_input = TimeDistributed(Flatten())(td_2)
        # transformer_output = TransformerEncoder(intermediate_dim=4,
        #                                         num_heads=3,
        #                                         activation='leaky_relu')(transformer_input)

        n_dense_weights = self.data_shape[0] * self.data_shape[1]**2 + self.data_shape[1]
        l1_factor = self.l1_magnitude/n_dense_weights

        dense_output = Dense(self.data_shape[1],
                             activation='sigmoid',
                             kernel_regularizer=keras.regularizers.L1L2(l1=l1_factor))(Flatten()(transformer_input))
        model = Model(inputs=node_input, outputs=dense_output)
        print(model.summary())

        # The following gives ~25% test precision
        """
        node_input = Input(shape=self.data_shape, name='node_input')
        node_feature_extractor = Conv1D(filters=1, kernel_size=1, activation='leaky_relu')
        td_output = Flatten()(TimeDistributed(node_feature_extractor)(node_input))
        dense_output = Dense(self.data_shape[1],
                             activation='sigmoid')(td_output)
        model = Model(inputs=node_input, outputs=dense_output)
        print(model.summary())
        """
        # The following gives ~30% test precision, ~75% training precision
        """
        node_input = Input(shape=self.data_shape, name='node_input')
        node_feature_extractor = Conv1D(filters=32, kernel_size=3, activation='relu')
        td_out_1 = TimeDistributed(node_feature_extractor)(node_input)
        td_out = TimeDistributed(Flatten())(td_out_1)
        lstm_out = Bidirectional(LSTM(64, return_sequences=False))(td_out)
        dense_1 = Dense(128)(lstm_out)
        dense_2 = Dense(64)(dense_1)
        dense_out = Dense(self.data_shape[1], activation='sigmoid')(dense_2)
        model = Model(inputs=node_input, outputs=dense_out)
        print(model.summary())
        """
        # The following gives ~36% test precision
        """
        node_input = Input(shape=self.data_shape, name='node_input')
        node_feature_extractor = Conv1D(filters=16, kernel_size=1, activation='relu')
        td_out_1 = TimeDistributed(node_feature_extractor)(node_input)
        td_out = TimeDistributed(Flatten())(td_out_1)
        lstm_out = Bidirectional(LSTM(32, return_sequences=False))(td_out)
        dense_1 = Dense(64, activation='leaky_relu')(lstm_out)
        dense_out = Dense(self.data_shape[1], activation='sigmoid', kernel_regularizer='l1')(dense_1)
        model = Model(inputs=node_input, outputs=dense_out)
        print(model.summary())
        """
        """
        node_input = GaussianNoise(1e-2)(Input(shape=self.data_shape, name='node_input'))
        node_feature_extractor = Conv1D(filters=4, kernel_size=1, activation='relu')
        td_out_1 = TimeDistributed(node_feature_extractor)(node_input)
        td_out = TimeDistributed(Flatten())(td_out_1)
        lstm_out = LSTM(self.data_shape[1], return_sequences=False)(td_out)
        dense_out = Dense(self.data_shape[1], activation='sigmoid')(lstm_out)
        model = Model(inputs=node_input, outputs=dense_out)
        print(model.summary())
        """
        """
        node_input = Input(shape=self.data_shape, name='node_input')
        # node_input_1 = LayerNormalization(axis=-2)(node_input)
        # node_input_2 = GaussianNoise(1e-2)(node_input_1)
        node_feature_extractor_1 = Conv1D(filters=6, kernel_size=1, activation='leaky_relu')
        node_feature_extractor_2 = Conv1D(filters=1, kernel_size=1, activation='sigmoid')
        td_out_1 = TimeDistributed(node_feature_extractor_1)(node_input)
        td_out_2 = TimeDistributed(node_feature_extractor_2)(node_input)
        td_out = TimeDistributed(Flatten())(td_out_2)
        dense_out = Dense(self.data_shape[1], activation='sigmoid', kernel_regularizer='l1')(Flatten()(td_out))
        # lstm_out = SimpleRNN(29, activation='sigmoid')(td_out)
        # dense_1 = Dense(self.data_shape[1])(lstm_out)
        # dense_out = Dense(self.data_shape[1], activation='sigmoid', kernel_regularizer=keras.regularizers.L1(l1=1e-4))(lstm_out)
        model = Model(inputs=node_input, outputs=Flatten()(td_out))
        print(model.summary())
        keras.utils.plot_model(model)
        """
        # TimeDistributed layer with Dense
        # time_output = TimeDistributed(Dense(units=self.node_embedding_length, activation='relu'))(node_input)
        # time_output = TransformerEncoder(intermediate_dim=8, num_heads=3, activation='leaky_relu')(node_input)
        # time_output = TimeDistributed(Dense(units=self.data_shape[1],
        #                                     kernel_regularizer=keras.regularizers.L1L2(l1=0.00, l2=0.00),
        #                                     activation='leaky_relu'))(normalised_input)
        # time_output = Flatten()(time_output)

        # Dense layer for the final output
        # dense_output = Dense(self.dense_units, kernel_regularizer=keras.regularizers.L1L2(l1=0.00, l2=0.00), activation='sigmoid')(time_output)
        # dense_output = Dense(self.dense_units)(time_output)

        # Create the Model
        # model = Model(inputs=node_input, outputs=dense_output)
        return model
    
    def compile(self):
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        metric = keras.metrics.Precision(top_k=1)
        self.model.compile(loss=self.custom_loss, optimizer=optimizer, metrics=['precision'], run_eagerly=True)

    def just_predict(self, train_x, train_y, observation_size, time_horizon, attack_window):

        print("Evaluating performance...")

        precision_log = []
        extra_precision_log = []

        m = keras.metrics.Precision()
        tp = keras.metrics.TruePositives()
        tn = keras.metrics.TrueNegatives()
        fp = keras.metrics.FalsePositives()
        fn = keras.metrics.FalseNegatives()

        for i in range(observation_size + attack_window, time_horizon):
            latest_obs = train_x[i - (observation_size-1):i + 1]

            predictions = self.model(tf.expand_dims(latest_obs, 0))[0]

            m.update_state(train_y[i], predictions)
            tp.update_state(train_y[i], predictions)
            tn.update_state(train_y[i], predictions)
            fp.update_state(train_y[i], predictions)
            fn.update_state(train_y[i], predictions)

            if np.max(predictions) >= 0.5:
                if train_y[i, np.argmax(predictions)] == 1:
                    precision_log.append(1)
                    extra_precision_log.append(np.max(predictions))
                else:
                    precision_log.append(0)
                    extra_precision_log.append(0)
            else:
                precision_log.append(np.nan)

        extra_precision_log = np.array(extra_precision_log)
        precision_log = np.array(precision_log)

        print(m.result())
        print(tp.result(), tn.result(), fp.result(), fn.result())
        print(np.mean(precision_log[np.isfinite(precision_log)]))
        print(len(precision_log[np.isfinite(precision_log)]))

        return np.array([np.mean(precision_log[np.isfinite(precision_log)]), m.result(), tp.result(), tn.result(), fp.result(), fn.result()])

    def evaluate_and_predict(self, train_x, train_y, observation_size, time_horizon, attack_window):

        print("Training...")

        rng = np.random.default_rng(seed=1234)
        n_nodes = train_y.shape[1]

        # replay_buffer_x_cutoff = observation_size + attack_window - 1

        replay_buffer_y = np.full(shape=train_y.shape, dtype=int, fill_value=np.nan)
        predictions_log = np.full(shape=train_y.shape, fill_value=np.nan)

        custom_precision_log = np.full(shape=train_y.shape[0], fill_value=np.nan)
        loss_log = []
        loss_stderr_log = []
        best_precision_log = []
        attack_p_log = []
        p_remaining_log = []
        remaining_steps_tracker = []

        extra_precision_log = []

        latest_predictions_log = np.full(shape=train_y.shape, fill_value=np.nan)

        times_sampled = np.ones(shape=time_horizon)

        armed = False

        for i in range(observation_size+attack_window, time_horizon):

            if i % 100 == 0:
                print(i)

            # replay_buffer_x_cutoff += 1
            replay_buffer_x = train_x[:i]

            # Update observed labels after node visit or after enough time has passed
            replay_buffer_y[:i-attack_window, :] = train_y[:i-attack_window, :]

            # Sample a batch from replay buffer
            batch_size = min(i - (attack_window+observation_size-1), self.batch_size)
            p = 1/times_sampled[observation_size-1:i-attack_window]**2 / np.sum(1/times_sampled[observation_size-1:i-attack_window]**2)
            indices = rng.choice(np.arange(observation_size, i - attack_window + 1)-1, size=batch_size, replace=False, p=p)
            batch_x = train_x[[np.arange(idx-observation_size, idx)+1 for idx in indices]]
            batch_y = train_y[[idx for idx in indices]]
            loss = self.model.train_on_batch(batch_x, batch_y)
            predictions = self.model.predict_on_batch(batch_x)

            self.positives += np.sum(train_y[i-attack_window])
            self.negatives += train_y[i-attack_window].shape[0] - np.sum(train_y[i-attack_window])

            for j, idx in enumerate(indices):

                times_sampled[idx] += 1

                predictions_log[idx] = predictions[j]
                max_ndx = np.argmax(predictions[j])

                if predictions[j, max_ndx] >= 0.5:
                    if batch_y[j, max_ndx] == 1:
                        custom_precision_log[idx] = 1
                    else:
                        custom_precision_log[idx] = 0
                else:
                    custom_precision_log[idx] = np.nan

            best_precision_log.append(np.mean(custom_precision_log[np.isfinite(custom_precision_log)]))
            loss_log.append(loss[0])

            p_attack = np.mean((predictions_log[np.isfinite(predictions_log).all(axis=1)] >= 0.5).any(axis=1))
            attack_p_log.append(p_attack)
            remaining_steps_tracker.append(time_horizon-attack_window-i)
            p_window_from_now = max(1 - (1-p_attack)**(time_horizon-attack_window-i), 0)
            p_remaining_log.append(p_window_from_now)

            if ((1-p_window_from_now)*10 > (1-best_precision_log[-1]) and not armed) or i == time_horizon - attack_window:
                armed = True
            #
            if armed:
                latest_obs = train_x[i-observation_size+1:i+1]
                current_predictions = self.model(tf.expand_dims(latest_obs, 0)).numpy()[0]
                print(current_predictions)
                print(train_y[i])
                if np.max(current_predictions) >= 0.5:
                    print(f"Attacking at timestep {i} at node {np.argmax(current_predictions)}")
                    if train_y[i, np.argmax(current_predictions)] == 1:
                        print("Success!")
                    else:
                        print("Failure!")
                    break

            # current_prediction = self.model(latest_obs)

        # extra_precision_log = np.array(extra_precision_log)
        # print(np.mean(extra_precision_log[np.isfinite(extra_precision_log)]))
        # print(np.mean(custom_precision_log[np.isfinite(custom_precision_log)]))
        # print(len(custom_precision_log[np.isfinite(custom_precision_log)]))

        # print(extra_precision_log[:10])
        # print(custom_precision_log[400:410])

        # for j in range(400, 410):
        #     print(replay_buffer_y[j])
        #     print(train_y[j])

        # print(predictions_log[500:510])
        # print(latest_predictions_log[500:510])
        # print(train_y[500:510])

        # plt.plot(best_precision_log)
        # plt.plot(attack_p_log)
        # plt.plot(loss_log)
        # plt.plot(np.arange(25, len(loss_log)), [np.std(loss_log[i-25:i])/np.mean(loss_log[i-25:i]) for i in np.arange(25, len(loss_log))])
        # plt.plot([np.std(loss_log[:j])/(j**0.5) for j in range(len(loss_log))])
        # plt.plot(p_remaining_log)
        # plt.show()

