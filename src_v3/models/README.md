# How to use the optimization script
1) Put your cost function in Autotuning-NGD/src/scream/cost_functions

2) Edit Autotuning-NGD/src/scream/surrogate_and_optimization/config_optimization.yaml to include the file name containing your cost function, the desired save directory, the directory containing the surrogate, and other details 

3) Grab a compute node using the following command:

salloc -A cli115 -N 1 -t 01:00:00

4) activate the autotuning environment (setup instructions in Autotuning-NGD/requirements_frontier.txt):

conda activate autotuning

5) Activate OpenBLAS multithreading to speed up computation (optional):

export OPENBLAS_NUM_THREADS=1

6) In the Autotuning-NGD/src/scream/surrogate_and_optimization directory, execute the following command (which takes on the order of 10 minutes to run):

python optimization.py config_optimization.yaml

# Read comments in the config file for further details, and read the optimization script for even more details
