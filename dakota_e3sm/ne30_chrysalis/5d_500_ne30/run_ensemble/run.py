#!/usr/bin/env python
"""
# $1 is params.in.(fn_eval_num) from Dakota
# $2 is results.out.(fn_eval_num) returned to Dakota

# To test:  ./run_and_postprocess.py workdir.1/params.in.1 resultsTest.out  
#     (but not all file references will work when looking up/down a directory)

# To use:   	- ON CEE:  module load percept/anaconda2
                - ON HPC:  modele load anaconda
"""
import os
import sys
import time
import shutil
#import numpy as np     
import pdb

# ---- Print info about parameters passed in


params_file_name  = sys.argv[1]
results_file_name = sys.argv[2]

print("  Params file = ", params_file_name, "\n")
print( "  Results file = ", results_file_name, "\n")

pwd = os.getcwd() 
print(pwd)

# --------------
# DAKOTA PRE-PROCESSING
# --------------
print( "  Running dprepro to create input file  ... \n")
cmd = 'dprepro --inline "{{ }}" ' + params_file_name + ' e3sm-inp.yml.template e3sm-inp.yaml'
#print(cmd) 
os.system(cmd)

# --------
# BUILD E3SM 
# --------

print( '\n  Building E3SM job ... \n')
os.system('./create.csh') # create, configure, build E3SM. 

#while not os.path.exists('./all_done.txt') :   
#    time.sleep(2)

# Change this to grep for 
#grep 'case.run success' CaseStatus in case_scripts. 

# ---------------
# POST-PROCESSING
# ---------------
print( '\n  No post processing --BMW  ... \n')
f = open("results.out", "w")
f.write(str(1.0))
f.close()

##-----File-Marshaling
pwd=os.getcwd()
results_file_name2 = pwd + '/' + results_file_name
shutil.copy('results.out', results_file_name2) 
print( results_file_name2)



