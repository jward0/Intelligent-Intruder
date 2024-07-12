#!/bin/bash

python3 main.py -s "sebs" -m "example" -n 6 -d 50 100 150 -t 600 1800 -b 1 2 4 8 -p 0.25 0.5 1.0 2.0 -l 0.0
python3 main.py -s "dtag" -m "cumberland" -n 2 -d 50 100 150 -t 600 1800 -b 1 2 4 8 -p 0.25 0.5 1.0 2.0 -l 0.0
python3 main.py -s "cbls" -m "DIAG_floor1" -n 4 -d 50 100 150 -t 600 1800 -b 1 2 4 8 -p 0.25 0.5 1.0 2.0 -l 0.0

