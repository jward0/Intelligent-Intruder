"""

Python script to be used in shell scripts

"""

# Python script: main.py
import argparse
# from Intruder_functionality import machine_learning as ml
import machine_learning as ml
import pandas as pd
import numpy as np
import os


# def get_dataset_results(strategy, map_, n_agents, trial_no, attack_window, time_horizon, l1_magnitude=0.1, batch_size=4, pos_weight=0.25, hidden_size=6):
#
#     extension = f"../datasets/{strategy}/{map_}/{n_agents}_agent{'s' if n_agents > 1 else ''}/{trial_no}/"
#
#     idle_data = np.genfromtxt(extension + "idleness.csv", delimiter=';')[:, 1:]
#     dist_data = np.genfromtxt(extension + "distance_metrics.csv", delimiter=';')[:, 1:]
#     vel_data = np.genfromtxt(extension + "velocity_metrics.csv", delimiter=';')[:, 1:]
#     vuln_data = np.genfromtxt(extension + "vulnerabilities.csv", delimiter=';')[:, 1:]
#
#     if strategy == "DTAP" and map_ == "DIAG_floor1" and n_agents == 1:
#         idle_data = idle_data[:19000]
#         dist_data = dist_data[:19000]
#         vel_data = vel_data[:19000]
#         vuln_data = vuln_data[:19000]
#
#     if strategy == "CBLS" and map_ == "DIAG_floor1" and n_agents == 1:
#         idle_data = idle_data[12000:]
#         dist_data = dist_data[12000:]
#         vel_data = vel_data[12000:]
#         vuln_data = vuln_data[12000:]
#
#     n_nodes = vel_data.shape[1]
#
#     idle_data[idle_data < 0] = np.nan  # set negative idle values to NaN for later removal
#
#     data = np.stack((idle_data/attack_window, dist_data, vel_data, vuln_data), axis=-1)
#     data = data[np.isfinite(data).all(axis=2).all(axis=1), :, :]  # Trim inf and NaN at start and end of data
#
#     train_x = data[:, :, :3]
#     train_y = (data[:, :, -1] >= attack_window).astype(int)
#
#     # Define the window size and thresholds
#     observation_size = 10
#
#     data_shape = (observation_size, train_x.shape[1], train_x.shape[-1])
#
#     n_tests = 4
#     repeats_per_data = 1
#
#     outcomes = np.zeros(shape=(n_tests, 8))
#
#     spacing = int((train_x.shape[0] - time_horizon)/n_tests)
#
#     for i in range(n_tests):
#
#         train_x_n = train_x[i*spacing:]
#         train_y_n = train_y[i*spacing:]
#
#         # if i > 0:
#         #     train_x = train_x[time_horizon:, :, :]
#         #     train_y = train_y[time_horizon:, :]
#
#         for _ in range(repeats_per_data):
#             # Reset and compile the model
#             model = ml.ML_Intruder(data_shape, n_nodes, l1_magnitude=l1_magnitude, batch_size=batch_size, pos_weight=pos_weight, hidden_size=hidden_size)
#             model.compile()
#
#             success = model.evaluate_and_predict(train_x_n, train_y_n, observation_size, time_horizon, attack_window)
#             results = model.just_predict(train_x[time_horizon:], train_y[time_horizon:], observation_size, time_horizon, attack_window)
#
#             outcomes[i] = np.insert([*results, success], 0, i)
#
#     return outcomes
#
#     # Output results to a csv
#
#     target_dir = f"results/{strategy}/{map_}/{n_agents}_agent{'' if n_agents == 1 else 's'}"
#     if not os.path.exists(target_dir):
#         os.makedirs(target_dir)
#
#     filename = target_dir + f"/{time_horizon}"
#
#     # print(outcomes)
#
#     np.savetxt(filename,
#                outcomes,
#                delimiter=',',
#                header="best_precision,overall_precision,true_positives,true_negatives,false_positives,false_negatives,success")


def get_dataset_results(data, attack_window, time_horizon, n_tests, model_params):

    n_nodes = data.shape[1]
    train_x = data[:, :, :3]
    train_y = (data[:, :, -1] >= attack_window).astype(int)

    # Define the window size and thresholds
    observation_size = 10

    data_shape = (observation_size, train_x.shape[1], train_x.shape[-1])

    outcomes = np.zeros(shape=n_tests)

    spacing = int((train_x.shape[0] - time_horizon)/n_tests)

    for i in range(n_tests):

        train_x_n = train_x[i*spacing:]
        train_y_n = train_y[i*spacing:]

        # Reset and compile the model
        model = ml.ML_Intruder(data_shape,
                               n_nodes,
                               l1_magnitude=model_params['l1'],
                               batch_size=model_params['bs'],
                               pos_weight=model_params['pw'],
                               hidden_size=model_params['hs'])
        model.compile()

        success = model.evaluate_and_predict(train_x_n, train_y_n, observation_size, time_horizon, attack_window)
        # results = model.just_predict(train_x[time_horizon:], train_y[time_horizon:], observation_size, time_horizon, attack_window)

        outcomes[i] = success

    return outcomes


def main(strategy, map_, n_agents, attack_windows, time_horizons, l1_magnitude=0.1, batch_size=4, pos_weight=0.25, hidden_size=6):

    runs = [0, 1, 2, 3, 4]

    model_params = {'l1': l1_magnitude, 'bs': batch_size, 'pw': pos_weight, 'hs': hidden_size}
    n_tests = 10

    extension = f"{strategy}/{map_}/{n_agents}_agent{'s' if n_agents > 1 else ''}"
    if not os.path.exists(f"../datasets/results/{extension}"):
        os.makedirs(f"../datasets/results/{extension}")

    th_results = [np.zeros(shape=len(attack_windows)) for _ in range(len(time_horizons))]

    for run in runs:

        idle_data = np.genfromtxt(f"../datasets/{extension}/{run}/idleness.csv", delimiter=';')[:, 1:]
        dist_data = np.genfromtxt(f"../datasets/{extension}/{run}/distance_metrics.csv", delimiter=';')[:, 1:]
        vel_data = np.genfromtxt(f"../datasets/{extension}/{run}/velocity_metrics.csv", delimiter=';')[:, 1:]
        vuln_data = np.genfromtxt(f"../datasets/{extension}/{run}/vulnerabilities.csv", delimiter=';')[:, 1:]

        vuln_data[np.isinf(vuln_data)] = 9999

        if strategy == "DTAP" and map_ == "DIAG_floor1" and n_agents == 1:
            idle_data = idle_data[:19000]
            dist_data = dist_data[:19000]
            vel_data = vel_data[:19000]
            vuln_data = vuln_data[:19000]

        if strategy == "CBLS" and map_ == "DIAG_floor1" and n_agents == 1:
            idle_data = idle_data[12000:]
            dist_data = dist_data[12000:]
            vel_data = vel_data[12000:]
            vuln_data = vuln_data[12000:]

        idle_data[idle_data < 0] = np.nan  # set negative idle values to NaN for later removal

        for (th_ndx, time_horizon) in enumerate(time_horizons):

            for (aw_ndx, attack_window) in enumerate(attack_windows):
                data = np.stack((idle_data / attack_window, dist_data, vel_data, vuln_data), axis=-1)
                data = data[np.isfinite(data).all(axis=2).all(axis=1), :, :]  # Trim inf and NaN at start and end of data

                outcomes = get_dataset_results(data, attack_window, time_horizon, n_tests, model_params)
                th_results[th_ndx][aw_ndx] += np.mean(outcomes) / len(runs)

                np.savetxt(f"../datasets/results/{extension}/{time_horizon}_{attack_window}_{run}.csv", outcomes, delimiter=',')

    for (th_ndx, time_horizon) in enumerate(time_horizons):
        np.savetxt(f"../datasets/results/{extension}/{time_horizon}.csv", th_results[th_ndx], delimiter=',')
        np.save(f"../datasets/results/{extension}/{time_horizon}.npy", th_results[th_ndx])


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
    args.add_argument(
        "-b", "--batch_size",
        nargs="*",
        type=int,
        default=[],
    )

    args.add_argument(
        "-p", "--pos_weight",
        nargs="*",
        type=float,
        default=[],
    )
    args.add_argument(
        "-l", "--l1_magnitude",
        nargs="*",
        type=float,
        default=[],
    )
    args.add_argument(
        "-hs", "--hidden_size",
        nargs="*",
        type=int,
        default=[],
    )

    a = args.parse_args()

    strategies = a.strategies
    maps = a.maps
    na = a.n_agents
    ad = a.attack_duration
    th = a.time_horizon
    bs = a.batch_size
    pw = a.pos_weight
    lm = a.l1_magnitude
    hs = a.hidden_size

    for s in strategies:
        for m in maps:
            for n in na:
                main(s, m, n, ad, th, lm[0], bs[0], pw[0], hs[0])

    # for l in lm:
    #     for b in bs:
    #         for p in pw:
    #             for a in ad:
    #                 for t in th:
    #                     for s in strategies:
    #                         for m in maps:
    #                             for n in na:
    #                                 for h in hs:
    #                                     for run in [0, 1, 2, 3, 4]:
    #                                         main(s, m, n, run, a, t, l, b, p, h)
