# AutoCalibration

OpenSource tools from the E3SM Autotuning team

This is a working repository for the "Autotuning" Epic, started in FY21 under the NGD Software and Algorithms portfolio. The project is working to add some automation to the GCM tuning process using UQ and ML algorithms.

POCs are Benjamin M. Wagman, Lyndsay Shand, and Andy Salinger (formerly Drew Yarger and K. Chowdhary). 


* The file "requirements.txt" sets up the Python packages and environment that we use for autotuning, resulting in the package versions described by "autotuning_environment.yaml". Our code depends on the "tesuract" and "clif" packages built by K. Chowdhary as specified in "requirements.txt". 

* The "dakota_e3sm/" folder contains the code and configuration for running the perturbed parameter ensembles (PPEs) in our work. 

* "src_v2/data/": preprocessed data used in Yarger et al. (2024) -- time-averaged spatial fields from the PPE, observations, and miscellaneous model runs

* The "src_v2/" folder contains code for the surrogate construction, optimization, and comparison of autotuning and default parameter sets that correspond to Yarger et al (2024) using E3SMv2. 
	* "src_v2/models/": in "surrogate_latest.py", code that creates surrogate and optimizes parameters, depending on code in "preprocessing.py", "postprocessing.py", and "optimization.py". The file "config_latest.yaml" contains parameters for the surrogate (for example, which fields to use) so that one can construct a surrogate and optimize parameters using "python surrogate_latest.py config_latest.yaml".
	* "src_v2/mcmc/": code that runs Markov Chain Monte Carlo based on the optimization function set up in "config_mcmc.yaml", so that one can construct a surrogate, optimize parameters, and run MCMC using "python mcmc.py config_mcmc.yaml". 
	* "src_v2/eda/": code for exploratory data analysis of the surrogate, perturbed parameter ensemble, and optimization results. This folder contains scripts to process the results after creating a surrogate, and is mostly based in R instead of Python. 
		* "src/eda/paper_plots": code that recreates figures in Yarger et al. (2024) 

* The "src_v3/" folder contains code for the surrogate construction, optimization, and figure generation that correspond to Wagman et al (2026) using E3SMv3. 
	* "src_v3/models/": in "surrogate_latest.py", code that creates surrogate and optimizes parameters, depending on code in "preprocessing.py", "postprocessing.py", and "optimization.py". The file "config_latest.yaml" contains parameters for the surrogate (for example, which fields to use) so that one can construct a surrogate and optimize parameters using "python surrogate_latest.py config_latest.yaml". 
		* Current config_surrogate.yaml and config_optimization.yaml configurations correspond to Wagman et al. (2026) and are set up to run with the following data on zenodo:
	* "src_v3/models/": in "surrogate_latest.py", code that creates surrogate and optimizes parameters, depending on code in "preprocessing.py", "postprocessing.py", and "optimization.py". The file "config_latest.yaml" contains parameters for the surrogate (for example, which fields to use) so that one can construct a surrogate and optimize parameters using "python surrogate_latest.py config_latest.yaml".
	* "src_v3/mcmc/": code that runs Markov Chain Monte Carlo based on the optimization function set up in "config_mcmc.yaml", so that one can construct a surrogate, optimize parameters, and run MCMC using "python mcmc.py config_mcmc.yaml". 
	* "src_v3/history matching/": code that runs history matching -like sampling procedure or additional sampling described in Section 2.2 of Wagman et al (2026).
	* "src/eda/": code for exploratory data analysis (EDA) of the surrogate, perturbed parameter ensemble, and optimization results. This folder contains scripts to process the results after creating a surrogate and miscellaneous data used in EDA. 


* As is currently set up, the processed data is expected to be placed in "data/", and the results from the autotuning procedure are expected to be placed in the "surrogate_models/" folder. 

* Using the preprocessed data found on zeonodo (add link), can be used to fit the surrogate and run the optimization code:

Step 1. Fit surrogate: python surrogate.py config_surrogate.yaml
Step 2. Run optimization: python optimization.py config_optimization.yaml

Yarger, D., Wagman, B. M., Chowdhary, K., & Shand, L. (2024). Autocalibration of the E3SM version 2 atmosphere model using a PCA-based surrogate for spatial fields. Journal of Advances in Modeling Earth Systems, 16, e2023MS003961. https://doi.org/10.1029/2023MS003961

Wagman, B. M., Collins, G., Shand, L. and Harrop, B. E. (2026). Calibration of E3SMv3 for Low or High Equilibrium Climate Sensitivity using a Machine Learning Surrogate Model. Under Review.





