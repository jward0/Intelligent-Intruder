"""

Python script to be used in shell scripts

"""

# Python script: main.py
import argparse
# from Intruder_functionality import machine_learning as ml
import machine_learning as ml
import pandas as pd
import numpy as np


def main(strategy, map_, n_agents, trial_no, attack_window=50, time_horizon=600):

    extension = f"../datasets/{strategy}/{map_}/{n_agents}_agent{'s' if n_agents > 1 else ''}/{trial_no}/"

    idle_data = np.genfromtxt(extension + "idleness.csv", delimiter=';')[:, 1:]
    dist_data = np.genfromtxt(extension + "distance_metrics.csv", delimiter=';')[:, 1:]
    vel_data = np.genfromtxt(extension + "velocity_metrics.csv", delimiter=';')[:, 1:]
    vuln_data = np.genfromtxt(extension + "vulnerabilities.csv", delimiter=';')[:, 1:]

    n_nodes = vel_data.shape[1]

    idle_data[idle_data < 0] = np.nan  # set negative idle values to NaN for later removal

    data = np.stack((idle_data/attack_window, dist_data, vel_data, vuln_data), axis=-1)
    data = data[np.isfinite(data).all(axis=2).all(axis=1), :, :]  # Trim inf and NaN at start and end of data

    train_x = data[:, :, :3]
    train_y = (data[:, :, -1] >= attack_window).astype(int)

    # Define the window size and thresholds
    observation_size = 1
    f1_threshold = 0.9  # confidence in f1
    f2_threshold = 0.999  # same as f1 threshold for now
    f3_threshold = 0.1  # numerically tested constant

    data_shape = (observation_size, train_x.shape[1], train_x.shape[-1])

    times_of_attack = np.array([])
    nodes_attacked = np.array([])
    attack_outcomes = np.array([])

    for i in range(1):
        # Drop the first n timesteps from train_x and train_y
        # TODO: WHY?
        if i > 0:
            train_x = train_x[600:, :, :]
            train_y = train_y[600:, :]

        for _ in range(1):
            # Reset and compile the model
            model = ml.ML_Intruder(data_shape, n_nodes)
            model.compile()

            model.just_predict(train_x, train_y, observation_size, time_horizon, attack_window)
            model.evaluate_and_predict(train_x, train_y, observation_size, time_horizon, attack_window)
            model.just_predict(train_x[time_horizon:], train_y[time_horizon:], observation_size, time_horizon, attack_window)
            model.just_predict(train_x, train_y, observation_size, time_horizon, attack_window)

            # time_of_attack, node_attacked, attack_outcome = model.evaluate_and_predict(
            #     train_x,
            #     train_y,
            #     observation_window,
            #     f1_threshold,
            #     f2_threshold,
            #     f3_threshold,
            #     ending_timestep,
            #     vuln_data)
            # times_of_attack = np.append(times_of_attack, time_of_attack)
            # nodes_attacked = np.append(nodes_attacked, node_attacked)
            # attack_outcomes = np.append(attack_outcomes, attack_outcome)

    # Output results to a text file
    # output_file = 'results.txt'
    # with open(output_file, 'w') as f:
    #     f.write('{}\n'.format(';'.join(map(str, times_of_attack))))
    #     f.write('{}\n'.format(';'.join(map(str, nodes_attacked))))
    #     f.write('{}\n'.format(';'.join(map(str, attack_outcomes))))
        

if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument(
        "-s", "--strategies",
        nargs="*",
        type=str,
        default=[],
    )
    args.add_argument(
        "-m", "--maps",
        nargs="*",
        type=str,
        default=[],
    )
    args.add_argument(
        "-n", "--n_agents",
        nargs="*",
        type=int,
        default=[],
    )

    args.add_argument(
        "-d", "--attack_duration",
        nargs="*",
        type=int,
        default=[],
    )
    args.add_argument(
        "-t", "--time_horizon",
        nargs="*",
        type=int,
        default=[],
    )

    a = args.parse_args()

    strategies = a.strategies
    maps = a.maps
    na = a.n_agents
    ad = a.attack_duration[0]
    th = a.time_horizon[0]

    for s in strategies:
        for m in maps:
            for n in na:
                for run in [0]:
                    main(s, m, n, run, ad, th)
