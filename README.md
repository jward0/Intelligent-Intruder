# Intelligent-Intruder
TCML adversary model for multi-robot patrolling

## Usage

The launchpoint for running an adversary test is main.py, which takes arguments as follows:
`-s --strategies` strategies to be tested
`-m --maps` maps to be tested
`-n --n_agents` patrol team sizes to be tested
`-d --attack_duration` adversary attack durations to be tested
`-t --time_horizon` scenario time horizons to be tested
`-b -p -l -hs` various hyperparameters (stick to default values)

So eg.
`python3 main.py -s "er" -m "example" -n 1 2 4 8 12 -d 30 90 180 -t 300 1200 3600`
Would test against ER strategy on ``example" map for teamsizes of 1-12 agents, attack durations of 30, 90, and 180s, and time horizons of 300, 1200, and 3600s.

## Input data

The expected format and location of input data is as follows:

Input data will be searched for at `../datasets/strategy_name/map_name/n_agents/run_no(expects 0,1,2,3,4)/`.
Within this directory, it expects `distance_metrics.csv`, `idleness.csv`, `velocity_metrics.csv`, `vulnerabilities.csv`.

`distance_metrics.csv`: First column is timestamps, subsequent columns are distance metrics as described in the paper for each vertex of the patrol graph.
`idleness.csv`: First column is timestamps, subsequent columns are vertex idlenesses.
`velocity_metrics.csv`: First column is timestamps, subsequent columns are velocity metrics as described in the paper for each vertex of the patrol graph.
`vulnerabilities.csv`: First column is timestamp, subsequent columns are vulnerabilities (ie. time until next visit) for each vertex of the patrol graph.

Note that timestamp columns should be identical across files.

## Output data

Data is logged to `../datasets/results/strategy_name/map_name/n_agents/`.
Within this, outputs from a test run for a given time horizon are logged as `time_horizon.csv` and `time_horizon.npy`, each of which contains observed adversary success probability as it varies with the selected range of attack duration.
