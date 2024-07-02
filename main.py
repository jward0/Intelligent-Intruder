"""

Python script to be used in shell scripts

"""

# Python script: main.py
import argparse
# from Intruder_functionality import machine_learning as ml
import machine_learning as ml
import pandas as pd
import numpy as np


def main(file_paths, attack_window=50, ending_timestep=600):
        
    file_path, file_path2, file_path3, file_path4 = file_paths

    df_vel = pd.read_csv("../datasets/" + file_path, sep=';', header=None)
    df_idle = pd.read_csv("../datasets/" + file_path2, sep=';', header=None)
    df_dist = pd.read_csv("../datasets/" + file_path3, sep=';', header=None)
    df_vuln = pd.read_csv("../datasets/" + file_path4, sep=';', header=None)

    n = len(df_vel.T)-1  # number of nodes in the environment

    # remove timestep column
    vel_data = df_vel.iloc[:, 1:n+1]
    idle_data = df_idle.iloc[:, 1:n+1].copy()  # use copy to avoid modification warning
    idle_data[idle_data < 0] = np.nan  # set negative idle values to NaN for later removal
    vuln_data = df_vuln.iloc[:, 1:n+1]
    dist_data = df_dist.iloc[:, 1:n+1]

    # combine datasets 
    df = pd.concat([vel_data, idle_data, dist_data, vuln_data], axis=1)
    df = df.replace([np.inf, -np.inf], np.nan)  # set all inf values to NaN for later removal

    # Drop rows with NaN values
    df = df.dropna()

    test = df.iloc[:, n*3:n*4].copy()
    vuln_data = test.to_numpy()

    # attack length threshold to determine attack success and update to binary classification:
    for col in range(n*3, n*4):
        df.iloc[:, col] = (df.iloc[:, col] >= attack_window).astype(int)

    # convert data to array
    dataset = df.to_numpy()

    # split into features and labels
    train_x, train_y = dataset[:, 0:-n], dataset[:, n*3:n*4]

    # reshape data to appropriate format as (timestep, number of nodes, number of features)
    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[-1]//3, 3))

    # Define the window size and thresholds
    window_size = 10
    f1_threshold = 0.999  # confidence in f1
    f2_threshold = 0.999  # same as f1 threshold for now
    f3_threshold = 0.1  # numerically tested constant

    data_shape = (window_size, train_x.shape[1], train_x.shape[-1])

    times_of_attack = np.array([])
    nodes_attacked = np.array([])
    attack_outcomes = np.array([])

    for i in range(10):
        # Drop the first 1000 timesteps from train_x and train_y
        if i > 0:
            train_x = train_x[600:, :, :]
            train_y = train_y[600:, :]

        for _ in range(2):
            # Reset and compile the model
            model = ml.ML_Intruder(data_shape, n)
            model.compile()

            time_of_attack, node_attacked, attack_outcome = model.evaluate_and_predict(
                train_x,
                train_y,
                window_size,
                f1_threshold,
                f2_threshold,
                f3_threshold,
                ending_timestep,
                vuln_data)
            times_of_attack = np.append(times_of_attack, time_of_attack)
            nodes_attacked = np.append(nodes_attacked, node_attacked)
            attack_outcomes = np.append(attack_outcomes, attack_outcome)

    # Output results to a text file
    output_file = 'results.txt'
    with open(output_file, 'w') as f:
        f.write('{}\n'.format(';'.join(map(str, times_of_attack))))
        f.write('{}\n'.format(';'.join(map(str, nodes_attacked))))
        f.write('{}\n'.format(';'.join(map(str, attack_outcomes))))
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Intruder Detection Model')
    parser.add_argument('file_paths', metavar='FILE', type=str, nargs=4,
                        help='file paths for velocity_metrics, idleness_metrics, distance_metrics, and vulnerability_metrics CSV files')
    args = parser.parse_args()
    main(args.file_paths)


