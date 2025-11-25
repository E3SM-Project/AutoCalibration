#!/bin/bash -fe

# After the ensemble was done, Benj re-reran the control using this script because needed a different casename between test and ref for diagnostics. 

# Control run. Branches off the spinup at year 11. 
# Benj added Andrew Bradley's layout changes for performance 
# Note Benj overwrites the pelayout in case_setup. 
# E3SM Water Cycle v2 run_e3sm script template.

main() {

# For debugging, uncomment libe below
#set -x

# --- Configuration flags ----

# Machine
readonly MACHINE=chrysalis

# Simulation
readonly COMPSET="F2010"
readonly RESOLUTION="ne30pg2_oECv3"
readonly DESCRIPTOR="F2010.ne30pg2_oECv3_control"
readonly CASE_GROUP="F2010"

# Code and compilation
readonly CHECKOUT="20210813"
readonly BRANCH="37959275bf3384157264e45a8d9c7c43f2be1d56" # Master Aug 13 2021
readonly DEBUG_COMPILE=false

# Run options
readonly MODEL_START_TYPE="branch"  # initial, continue
readonly START_DATE="00011-01-01"
readonly STOP_OPTION=nyears
readonly STOP_N=10
readonly REST_OPTION=nyears
readonly REST_N=1
readonly DO_SHORT_TERM_ARCHIVING=false

# Coupler history 
readonly HIST_OPTION="nyears"
readonly HIST_N="1"

# Batch options
#readonly PELAYOUT='run_production_Custom_29'
#readonly PELAYOUT="M"
#readonly PELAYOUT="L"
readonly PELAYOUT=production_M
readonly PROJECT="e3sm"
readonly WALLTIME="28:00:00"
readonly TASKS=640

# Case name
#readonly CASE_NAME=${CHECKOUT}.${DESCRIPTOR}.${RESOLUTION}
readonly CASE_NAME=${CHECKOUT}.${DESCRIPTOR}.${MACHINE}

# Set paths
readonly CODE_ROOT="${HOME}/E3SM/code/${CHECKOUT}"
readonly CASE_ROOT="/lcrc/group/e3sm/${USER}/E3SM_simulations/dakota/5d_chr_500_ne30_10nodes/ctrl/${CASE_NAME}"

# Additional options for 'branch' and 'hybrid'
readonly GET_REFCASE=TRUE
readonly RUN_REFDIR="/lcrc/group/e3sm/ac.wagman/E3SM_simulations/dakota/5d_chr_1000_ne30/spinup/20210813.F2010.ne30pg2_oECv3.chrysalis/archive/rest/0011-01-01-00000/"
readonly RUN_REFCASE="20210813.F2010.ne30pg2_oECv3.chrysalis"
readonly RUN_REFDATE="0011-01-01"   # same as MODEL_START_DATE for 'branch', can be different for 'hybrid'

# Sub-directories (leave unchanged)
readonly CASE_BUILD_DIR=${CASE_ROOT}/build
readonly CASE_ARCHIVE_DIR=${CASE_ROOT}/archive
readonly CASE_SCRIPTS_DIR=${CASE_ROOT}/case_scripts
readonly CASE_RUN_DIR=${CASE_ROOT}/run

# Leave empty (unless you understand what it does)
#readonly OLD_EXECUTABLE="/lcrc/group/e3sm/ac.wagman/E3SM_simulations/dakota/5d_chr_1000_ne30/spinup/20210813.F2010.ne30pg2_oECv3.chrysalis/build/e3sm.exe"
readonly OLD_EXECUTABLE=""

# --- Toggle flags for what to do ----
do_fetch_code=false
do_create_newcase=true
do_case_setup=true
do_case_build=true
do_case_submit=false

# --- Now, do the work ---

# Make directories created by this script world-readable
#umask 022

# Fetch code from Github
fetch_code

# Create case
create_newcase

# Setup
case_setup # NOTE I have over-written the PE layout. Scroll down to case setup. 



# Build
case_build

# Configure runtime options
runtime_options

# Copy script into case_script directory for provenance
copy_script

# Submit
case_submit

# All done
echo $'\n----- All done -----\n'

}

# =======================
# Custom user_nl settings
# =======================

user_nl() {

cat << EOF >> user_nl_eam
 nhtfrq =   0,0
 mfilt  = 1,1
 avgflag_pertape = 'A','A'

 INITHIST='ENDOFRUN' 
 fincl1='Z500','RH500','T500','U850','U200'
 

! Additional retuning. Unnecessary because this is the new default. 
 clubb_tk1 = 268.15D0

EOF

cat << EOF >> user_nl_elm

EOF


}

######################################################
### Most users won't need to change anything below ###
######################################################

#-----------------------------------------------------
fetch_code() {

    if [ "${do_fetch_code,,}" != "true" ]; then
        echo $'\n----- Skipping fetch_code -----\n'
        return
    fi

    echo $'\n----- Starting fetch_code -----\n'
    local path=${CODE_ROOT}
    local repo=e3sm

    echo "Cloning $repo repository branch $BRANCH under $path"
    if [ -d "${path}" ]; then
        echo "ERROR: Directory already exists. Not overwriting"
        exit 20
    fi
    mkdir -p ${path}
    pushd ${path}

    # This will put repository, with all code
    git clone git@github.com:E3SM-Project/${repo}.git .
    
    # Setup git hooks
    rm -rf .git/hooks
    git clone git@github.com:E3SM-Project/E3SM-Hooks.git .git/hooks
    git config commit.template .git/hooks/commit.template

    # Bring in all submodule components
    git submodule update --init --recursive

    # Check out desired branch
    git checkout ${BRANCH}

    popd
}

#-----------------------------------------------------
create_newcase() {

    if [ "${do_create_newcase,,}" != "true" ]; then
        echo $'\n----- Skipping create_newcase -----\n'
        return
    fi

    echo $'\n----- Starting create_newcase -----\n'

    ${CODE_ROOT}/cime/scripts/create_newcase \
        --case ${CASE_NAME} \
        --case-group ${CASE_GROUP} \
        --output-root ${CASE_ROOT} \
        --script-root ${CASE_SCRIPTS_DIR} \
        --handle-preexisting-dirs u \
        --compset ${COMPSET} \
        --res ${RESOLUTION} \
        --machine ${MACHINE} \
        --project ${PROJECT} \
        --walltime ${WALLTIME} \
        --pecount ${PELAYOUT} 
    if [ $? != 0 ]; then
      echo $'\nNote: if create_newcase failed because sub-directory already exists:'
      echo $'  * delete old case_script sub-directory'
      echo $'  * or set do_newcase=false\n'
      exit 35
    fi

}
#--------20200415 Zheng: for theta-l + SL tracer transport
custom_pelayout() {
  tmp=($(echo $run | tr "_" " "))
  layout=${tmp[1]}
  nnodes=${tmp[2]}
  if [ "${layout}" == "Custom" ]; then
    echo $'\n CUSTOMIZE PROCESSOR CONFIGURATION:'
    pushd ${CASE_SCRIPTS_DIR}

    ntask=$((${nnodes} * 32)) 

#    ./xmlchange --id CAM_CONFIG_OPTS --append --val="-pcols $pcol"
    ./xmlchange NTASKS="$ntask"
    ./xmlchange ROOTPE="0"
    ./xmlchange NTHRDS="1"
    
	popd
 fi
}

#-----------------------------------------------------
case_setup() {

    if [ "${do_case_setup,,}" != "true" ]; then
        echo $'\n----- Skipping case_setup -----\n'
        return
    fi

    echo $'\n----- Starting case_setup -----\n'
    pushd ${CASE_SCRIPTS_DIR}

    # BEGIN BMW OVERWRITE
    ./xmlchange NTASKS=$TASKS #960 tasks = 30 nodes. 640=10 nodes
    ./xmlchange ROOTPE=0
    ./xmlchange NTHRDS=1
    # ABradley's suggestions below. 
    ./xmlchange MAX_MPITASKS_PER_NODE=64  # but then request 64 ranks/node FOR CHRYSALIS added by BMW at suggestion of ABradley
    ./xmlchange MAX_TASKS_PER_NODE=64  # redundant but to be absolutely clear about the (rank,thread) layout per node
    # END BMW OVERWRITE 


    # Setup some CIME directories
    ./xmlchange EXEROOT=${CASE_BUILD_DIR}
    ./xmlchange RUNDIR=${CASE_RUN_DIR}

    # Short term archiving
    ./xmlchange DOUT_S=${DO_SHORT_TERM_ARCHIVING^^}
    ./xmlchange DOUT_S_ROOT=${CASE_ARCHIVE_DIR}

    # Build with COSP, except for a data atmosphere (datm)
    if [ `./xmlquery --value COMP_ATM` == "datm"  ]; then 
      echo $'\nThe specified configuration uses a data atmosphere, so cannot activate COSP simulator\n'
    else
      echo $'\nConfiguring E3SM to use the COSP simulator\n'
      ./xmlchange --id CAM_CONFIG_OPTS --append --val='-cosp'
    fi

    # Extracts input_data_dir in case it is needed for user edits to the namelist later
    local input_data_dir=`./xmlquery DIN_LOC_ROOT --value`

    # Custom user_nl
    user_nl

    # Finally, run CIME case.setup
    ./case.setup --reset

    
    popd
}

#-----------------------------------------------------
case_build() {

    pushd ${CASE_SCRIPTS_DIR}

    # do_case_build = false
    if [ "${do_case_build,,}" != "true" ]; then

        echo $'\n----- case_build -----\n'

        if [ "${OLD_EXECUTABLE}" == "" ]; then
            # Ues previously built executable, make sure it exists
            if [ -x ${CASE_BUILD_DIR}/e3sm.exe ]; then
                echo 'Skipping build because $do_case_build = '${do_case_build}
            else
                echo 'ERROR: $do_case_build = '${do_case_build}' but no executable exists for this case.'
                exit 297
            fi
        else
            # If absolute pathname exists and is executable, reuse pre-exiting executable
            if [ -x ${OLD_EXECUTABLE} ]; then
                echo 'Using $OLD_EXECUTABLE = '${OLD_EXECUTABLE}
                cp -fp ${OLD_EXECUTABLE} ${CASE_BUILD_DIR}/
            else
                echo 'ERROR: $OLD_EXECUTABLE = '$OLD_EXECUTABLE' does not exist or is not an executable file.'
                exit 297
            fi
        fi
        echo 'WARNING: Setting BUILD_COMPLETE = TRUE.  This is a little risky, but trusting the user.'
        ./xmlchange BUILD_COMPLETE=TRUE

    # do_case_build = true
    else

        echo $'\n----- Starting case_build -----\n'

        # Turn on debug compilation option if requested
        if [ "${DEBUG_COMPILE^^}" == "TRUE" ]; then
            ./xmlchange DEBUG=${DEBUG_COMPILE^^}
        fi

        # Run CIME case.build
        ./case.build

        # Some user_nl settings won't be updated to *_in files under the run directory
        # Call preview_namelists to make sure *_in and user_nl files are consistent.
        ./preview_namelists

    fi

    popd
}

#-----------------------------------------------------
runtime_options() {

    echo $'\n----- Starting runtime_options -----\n'
    pushd ${CASE_SCRIPTS_DIR}

    # Set simulation start date
    ./xmlchange RUN_STARTDATE=${START_DATE}

    # Segment length
    ./xmlchange STOP_OPTION=${STOP_OPTION,,},STOP_N=${STOP_N}

    # Restart frequency
    ./xmlchange REST_OPTION=${REST_OPTION,,},REST_N=${REST_N}

    # Coupler history
    ./xmlchange HIST_OPTION=${HIST_OPTION,,},HIST_N=${HIST_N}

    # Coupler budgets (always on)
    ./xmlchange BUDGETS=TRUE

    # Set resubmissions
    if (( RESUBMIT > 0 )); then
        ./xmlchange RESUBMIT=${RESUBMIT}
    fi

    # Run type
    # Start from default of user-specified initial conditions
    if [ "${MODEL_START_TYPE,,}" == "initial" ]; then
        ./xmlchange RUN_TYPE="startup"
        ./xmlchange CONTINUE_RUN="FALSE"

    # Continue existing run
    elif [ "${MODEL_START_TYPE,,}" == "continue" ]; then
        ./xmlchange CONTINUE_RUN="TRUE"
	
    elif [ "${MODEL_START_TYPE,,}" == "branch" ] || [ "${MODEL_START_TYPE,,}" == "h
ybrid" ]; then
        ./xmlchange RUN_TYPE=${MODEL_START_TYPE,,}
        ./xmlchange GET_REFCASE=${GET_REFCASE}
        ./xmlchange RUN_REFDIR=${RUN_REFDIR}
        ./xmlchange RUN_REFCASE=${RUN_REFCASE}
        ./xmlchange RUN_REFDATE=${RUN_REFDATE}
        echo 'Warning: $MODEL_START_TYPE = '${MODEL_START_TYPE} 
        echo '$RUN_REFDIR = '${RUN_REFDIR}
        echo '$RUN_REFCASE = '${RUN_REFCASE}
        echo '$RUN_REFDATE = '${START_DATE}
 
    else
        echo 'ERROR: $MODEL_START_TYPE = '${MODEL_START_TYPE}' is unrecognized. Exiting.'
        exit 380
    fi

    # Patch mpas streams files
    #patch_mpas_streams

    popd
}



#-----------------------------------------------------
case_submit() {

    if [ "${do_case_submit,,}" != "true" ]; then
        echo $'\n----- Skipping case_submit -----\n'
        return
    fi

    echo $'\n----- Starting case_submit -----\n'
    pushd ${CASE_SCRIPTS_DIR}
    
    # Run CIME case.submit
    ./case.submit

    popd
}

#-----------------------------------------------------
copy_script() {

    echo $'\n----- Saving run script for provenance -----\n'

    local script_provenance_dir=${CASE_SCRIPTS_DIR}/run_script_provenance
    mkdir -p ${script_provenance_dir}
    local this_script_name=`basename $0`
    local script_provenance_name=${this_script_name}.`date +%Y%m%d-%H%M%S`
    cp -vp ${this_script_name} ${script_provenance_dir}/${script_provenance_name}

}

#-----------------------------------------------------
# Silent versions of popd and pushd
pushd() {
    command pushd "$@" > /dev/null
}
popd() {
    command popd "$@" > /dev/null
}

# Now, actually run the script
#-----------------------------------------------------
main


