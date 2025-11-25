#!/usr/bin/env python
"""
# $1 is params.in.(fn_eval_num) from Dakota
# $2 is results.out.(fn_eval_num) returned to Dakota

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
cmd = 'dprepro --no-warn --inline "{{ }}" ' + params_file_name + ' e3sm-inp.yml.template e3sm-inp.yaml'
cmdp = 'pyprepro --inline "{ }" ' + params_file_name + '  e3sm-inp.yml.template  e3sm-inp.yaml'
print(cmd) 
os.system(cmd)

# --------
# BUILD E3SM 
# --------

print( '\n  Setting up E3SM ... \n')
os.system(' python setup_e3sm.py') # create, configure, build E3SM. 
#print( '\n  Turned off setup_e3sm.py in run.py. For testing. ... \n')
#while not os.path.exists('./finished_e3sm_setup') :   
#    time.sleep(2)

# ---------------
# POST-PROCESSING
# ---------------
#print( '\n  No post processing --BMW  ... \n')
f = open("results.out", "w")
f.write(str(1.0))
f.close()

##-----File-Marshaling
pwd=os.getcwd()
results_file_name2 = pwd + '/' + results_file_name
shutil.copy('results.out', results_file_name2) 
print( results_file_name2)



