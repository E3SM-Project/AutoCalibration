# How to use these calibration scripts

1). Request a compute node: srun -N 1 -t 120 --pty bash

2) Run the following in python:

python surrogate_latest.py config_latest.yaml

This file loads the netcdf data in the /data/ directory and fits a data-driven ROM model and then calibrates using a deterministic optimization. Feel free to contact me with question as the code is not fully documented. 
