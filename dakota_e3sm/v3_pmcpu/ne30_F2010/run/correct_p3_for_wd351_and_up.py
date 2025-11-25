import os
import pdb
from glob import glob
from shutil import copyfile

# Apply the p3_mincdnc correction to workdirs > 350

root = '/pscratch/sd/w/wagmanbe/E3SM_simulations/dakota/v3_pmcpu/ne30_F2010/ens/'

workdirs = [a for a in os.listdir(root) if os.path.isdir(os.path.join(root,a))]
params_file_data = []

for wd in workdirs:
    dak_params_file = glob( os.path.join( root, wd, 'params.in.???'))
    user_nl_files    = glob( os.path.join( root, wd, '2023*','user_nl_eam')) 
    for f in dak_params_file + user_nl_files:
        wd_num = int( wd.split('.')[1] )
        if wd_num > 350:
            # copy the file to its original name + '_orig'
            f_orig = f+'_orig'
            f_new  = f+'_new'
            copyfile( f, f_orig)
            copyfile( f, f_new)
            # Multiply p3_mincdnc by 10^6
            infile = open( f, 'r')
            list_of_lines = infile.readlines()
            infile.close()
            for i in range(len(list_of_lines)):
                if 'p3_mincdnc' in list_of_lines[i]:
                    if 'params' in f:
                        if not 'DVV' in list_of_lines[i]:
                            params_file_data.append(float( list_of_lines[i].split("p3")[0].split("\n")[0]))
                    
                            #if 'e+06' in list_of_lines[i]:
                            #    list_of_lines[i] = list_of_lines[i].replace('e+06','e+07')
                    
                            if 'e+01' in list_of_lines[i]:
                                list_of_lines[i] = list_of_lines[i].replace('e+01','e+07')
                            if 'e+00' in list_of_lines[i]:
                                list_of_lines[i] = list_of_lines[i].replace('e+00','e+06')
                    else:
                        if 'user_nl' in f:
                            # Get the digits. Multiply by 1e6.
                            digits_string = list_of_lines[i].split("=")[1].split("\n")[0]
                            if float(digits_string) < 5e6:
                                new_digits_string = str(float(digits_string) * 1e6)
                                list_of_lines[i] = list_of_lines[i].replace(digits_string,new_digits_string)
                            
            # Now write list of lines to outfile.
            outfile = open( f_new,'w') # write over it
            outfile.writelines(list_of_lines)
            outfile.close()
            copyfile(f_new, f ) # Overwrite the original file. THe orig file data is preserved in ""_orig
            
