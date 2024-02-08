"""

Python script to be used in shell scripts

"""

# Python script: main.py
import argparse
from Intruder_functionality import machine_learning as ml
import pandas as pd
import numpy as np


def main(file_paths, attack_window = 50, ending_timestep = 2000):

    file_path, file_path2, file_path3, file_path4 = file_paths

    df_vel = pd.read_csv(file_path, sep=';', header=None)
    df_idle = pd.read_csv(file_path2, sep=';',header=None)
    df_vuln = pd.read_csv(file_path3, sep=';',header=None)
    df_dist = pd.read_csv(file_path4, sep=';',header=None)

    N = len(df_vel.T)-1 # number of nodes in the enviroment

    # remove timestep column
    vel_data = df_vel.iloc[:, 1:N+1]
    idle_data = df_idle.iloc[:, 1:N+1].copy() # use copy to avoid modifcation warning
    idle_data[idle_data < 0] = np.nan # set negative idle values to NaN for later removal
    vuln_data = df_vuln.iloc[:, 1:N+1]
    dist_data = df_dist.iloc[:, 1:N+1]

    # combine datasets 
    df = pd.concat([vel_data,idle_data,dist_data,vuln_data],axis=1)
    df = df.replace([np.inf, -np.inf], np.nan) # set all inf values to NaN for later removal

    # Drop rows with NaN values
    df = df.dropna()

    # attack length threshold to determine attack success and update to binary classification:
    for col in range(N*3,N*4):
        df.iloc[:, col] = (df.iloc[:, col] >= attack_window).astype(int)

    # convert data to array
    dataset = df.to_numpy()

    # split into features and labels
    trainX, trainY = dataset[:,0:-N], dataset[:,N*3:N*4] 

    # reshape data to appropriate format as (timestep, number of nodes, number of features)
    trainX = np.reshape(trainX, (trainX.shape[0],trainX.shape[-1]//3,3))

    # Define the window size and thresholds
    window_size = 10
    f1_threshold = 1  # confidence in f1
    f2_threshold = 1    # same as f1 threshold for now
    f3_thresholds = np.arange(0,1,0.05)     # numerically tested constant

    data_shape = (window_size,trainX.shape[1], trainX.shape[-1])
    model = ml.ML_Intruder(data_shape)

    # Compile the model
    model.compile()

    times_of_attack = np.array([])
    nodes_attacked = np.array([])
    attack_outcomes = np.array([])

    for f3_threshold in f3_thresholds:
    #
        time_of_attack, node_attacked, attack_outcome = model.evaluate_and_predict(trainX, trainY, window_size, f1_threshold, f2_threshold, f3_threshold, ending_timestep)
        times_of_attack = np.append(times_of_attack, time_of_attack)
        nodes_attacked = np.append(nodes_attacked, node_attacked)
        attack_outcomes = np.append(attack_outcomes, attack_outcome)

    # Output results to a text file
    with open('results.txt', 'w') as f:
        f.write('Times of Attack: {}\n'.format(times_of_attack))
        f.write('Nodes Attacked: {}\n'.format(nodes_attacked))
        f.write('Attack Outcomes: {}\n'.format(attack_outcomes))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Intruder Detection Model')
    parser.add_argument('file_paths', metavar='FILE', type=str, nargs=4,
                        help='file paths for velocity_metrics, idleness, vulnerabilities, and distance_metrics CSV files')
    args = parser.parse_args()
    main(args.file_paths)


