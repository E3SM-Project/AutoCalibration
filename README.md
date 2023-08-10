# AutoCalibration

OpenSource tools from the E3SM Autotuning team

This is a working repository for the "Autotuning" Epic, started in FY21 under the NGD Software and Algorithms portfolio. The project is working to add some automation to the GCM tuning process using UQ and ML algorithms.

POCs are Benj Wagman, Drew Yarger, Lyndsay Shand, and Andy Salinger (formerly K. Chowdhary). 

The slides from our March 2nd All-Hands Webinar presentation are at https://github.com/E3SM-Project/Autotuning-NGD/blob/main/docs/presentations/e3sm_allhands_2023/allhands_slides.pptx.

* The file "requirements.txt" sets up the Python packages and environment that we use for autotuning, resulting in the package versions described by "autotuning_environment.yaml". Our code depends on the "tesuract" and "clif" packages built by K. Chowdhary as specified in "requirements.txt". 

* The "dakota_e3sm/" folder contains the code and configuration for running the perturbed parameter ensembles (PPEs) in our work. 

* The "src/" folder contains code for the surrogate construction, optimization, and comparison of autotuning and default parameter sets. 
	* "src/data/": code that processes the time-averaged spatial fields from the PPE, observations, and miscellaneous model runs. By default, the resulting files are stored in "data/".
	* "src/models/": in "surrogate_latest.py", code that creates surrogate and optimizes parameters, depending on code in "preprocessing.py", "postprocessing.py", and "optimization.py". The file "config_latest.yaml" contains parameters for the surrogate (for example, which fields to use) so that one can construct a surrogate and optimize parameters using "python surrogate_latest.py config_latest.yaml".
	* "src/mcmc/": code that runs Markov Chain Monte Carlo based on the optimization function set up in "config_mcmc.yaml", so that one can construct a surrogate, optimize parameters, and run MCMC using "python mcmc.py config_mcmc.yaml". 
	* "src/eda/": code for exploratory data analysis of the surrogate, perturbed parameter ensemble, and optimization results. This folder contains scripts to process the results after creating a surrogate, and is mostly based in R instead of Python. 
	* "src/visualization/": code that creates visualizations of predicted fields for the surrogate. 


* As is currently set up, the processed data is expected to be placed in "data/", and the results from the autotuning procedure are expected to be placed in the "surrogate_models/" folder. 




