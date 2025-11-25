#!/bin/bash -fe

# E3SM Coupled Model Group run_e3sm script template.
#
# Bash coding style inspired by:
# http://kfirlavi.herokuapp.com/blog/2012/11/14/defensive-bash-programming

main() {

# For debugging, uncomment libe below
#set -x

# --- Configuration flags ----

# Machine and project
readonly MACHINE=chrysalis
readonly PROJECT="e3sm"    

# Simulation
readonly COMPSET="WCYCL1850"
readonly RESOLUTION="ne30pg2_r05_EC30to60E2r2"
readonly NL_MAPS=true   ### nonlinear maps for tri-grid
readonly CASE_NAME="20230924.v3alpha04_trigrid.piControl.chrysalis.opt.5e6"

# Code and compilation
readonly CHECKOUT="20230924"
readonly BRANCH="84e50561a854e1888b0eaa52fc3a44287f3a5924" # master as on 2023-09-14
readonly CHERRY=( )
readonly DEBUG_COMPILE=false

# Run options
readonly MODEL_START_TYPE="hybrid"  # 'initial', 'continue', 'branch', 'hybrid'
readonly START_DATE="0001-01-01"

# Additional options for 'branch' and 'hybrid'
readonly GET_REFCASE=TRUE
readonly RUN_REFDIR="/lcrc/group/e3sm2/ac.xzheng/E3SMv3_dev/20230924.v3alpha04_trigrid.piControl.chrysalis/init/"
readonly RUN_REFCASE="20221012.submeso.piControl.ne30pg2_EC30to60E2r2.chrysalis"  # new v3atm species appended to REFCASE's eam.i
readonly RUN_REFDATE="0301-01-01"

# Set paths
readonly CODE_ROOT="/home/${USER}/E3SM/code/${CHECKOUT}"
readonly CASE_ROOT="/lcrc/group/e3sm/${USER}/E3SM_simulations/v3/${CASE_NAME}"

# Sub-directories
readonly CASE_BUILD_DIR=${CASE_ROOT}/build
readonly CASE_ARCHIVE_DIR=${CASE_ROOT}/archive

# Define type of run
#  short tests: 'XS_1x10_ndays', 'XS_2x5_ndays', 'S_1x10_ndays', 'M_1x10_ndays', 'L_1x10_ndays'
#  or 'production' for full simulation

#readonly run='XS_2x5_ndays'
#readonly run='custom-100-production'
#readonly run='S_1x10_ndays'
readonly run='production'
if [[ "${run}" != *"production"* ]] ;then
  # Short test simulations
  tmp=($(echo $run | tr "_" " "))
  layout=${tmp[0]}
  units=${tmp[2]}
  resubmit=$(( ${tmp[1]%%x*} -1 ))
  length=${tmp[1]##*x}

  readonly CASE_SCRIPTS_DIR=${CASE_ROOT}/tests/${run}/case_scripts
  readonly CASE_RUN_DIR=${CASE_ROOT}/tests/${run}/run
  readonly PELAYOUT=${layout}
  readonly WALLTIME="2:00:00"
  readonly STOP_OPTION=${units}
  readonly STOP_N=${length}
  readonly REST_OPTION=${STOP_OPTION}
  readonly REST_N=${STOP_N}
  readonly RESUBMIT=${resubmit}
  readonly DO_SHORT_TERM_ARCHIVING=false

else

  # Production simulation
  readonly CASE_SCRIPTS_DIR=${CASE_ROOT}/case_scripts
  readonly CASE_RUN_DIR=${CASE_ROOT}/run
  readonly PELAYOUT="L"
  readonly WALLTIME="24:00:00"
  readonly STOP_OPTION="nyears"
  readonly STOP_N="20"
  readonly REST_OPTION="nyears"
  readonly REST_N="1"
  readonly RESUBMIT="9"
  readonly DO_SHORT_TERM_ARCHIVING=false
fi

# Coupler history 
readonly HIST_OPTION="nyears"
readonly HIST_N="1"

# Leave empty (unless you understand what it does)
readonly OLD_EXECUTABLE=""

# --- Toggle flags for what to do ----
do_fetch_code=false
do_create_newcase=true
do_case_setup=true
do_case_build=true
do_case_submit=true

# --- Now, do the work ---

# Make directories created by this script world-readable
umask 022

# Fetch code from Github
fetch_code

# Create case
create_newcase

# Custom PE layout
custom_pelayout

# Setup
case_setup

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
 cosp_lite = .true.

 empty_htapes = .true.

 avgflag_pertape = 'A','A','A','A','I','I'
 nhtfrq = 0,-24,-6,-3,-1,0
 mfilt  = 1,30,120,240,720,1

 fincl1 = 'AODALL','AODBC','AODDUST','AODPOM','AODSO4','AODSOA','AODSS','AODVIS',
          'CLDLOW','CLDMED','CLDHGH','CLDTOT',
          'CLDHGH_CAL','CLDLOW_CAL','CLDMED_CAL','CLD_MISR','CLDTOT_CAL',
          'CLMODIS','FISCCP1_COSP','FLDS','FLNS','FLNSC','FLNT','FLUT',
          'FLUTC','FSDS','FSDSC','FSNS','FSNSC','FSNT','FSNTOA','FSNTOAC','FSNTC',
          'ICEFRAC','LANDFRAC','LWCF','OCNFRAC','OMEGA','PRECC','PRECL','PRECSC','PRECSL','PS','PSL','Q',
          'QFLX','QREFHT','RELHUM','SCO','SHFLX','SOLIN','SWCF','T','TAUX','TAUY','TCO',
          'TGCLDLWP','TMQ','TREFHT','TREFMNAV','TREFMXAV','TS','U','U10','V','Z3'
          'SFbc_a1','SFbc_a3','SFbc_a4', 'SFdst_a1','SFdst_a3',
          'SFmom_a1','SFmom_a2','SFmom_a3','SFmom_a4',
          'SFncl_a1','SFncl_a2','SFncl_a3', 'SFpom_a1','SFpom_a3','SFpom_a4',
          'SFso4_a1','SFso4_a2','SFso4_a3','SFso4_a5', 'SFsoa_a1','SFsoa_a2','SFsoa_a3',
          'bc_a4_CLXF','pom_a4_CLXF','so4_a1_CLXF','so4_a2_CLXF',
          'bc_a1DDF','bc_a3DDF','bc_a4DDF','bc_c1DDF','bc_c3DDF','bc_c4DDF',
          'dst_a1DDF','dst_a3DDF','dst_c1DDF','dst_c3DDF',
          'mom_a1DDF','mom_a2DDF','mom_a3DDF','mom_a4DDF','mom_c1DDF','mom_c2DDF','mom_c3DDF','mom_c4DDF',
          'ncl_a1DDF','ncl_a2DDF','ncl_a3DDF','ncl_c1DDF','ncl_c2DDF','ncl_c3DDF',
          'pom_a1DDF','pom_a3DDF','pom_a4DDF','pom_c1DDF','pom_c3DDF','pom_c4DDF',
          'so4_a1DDF','so4_a2DDF','so4_a3DDF','so4_a5DDF','so4_c1DDF','so4_c2DDF','so4_c3DDF','so4_c5DDF',
          'soa_a1DDF','soa_a2DDF','soa_a3DDF','soa_c1DDF','soa_c2DDF','soa_c3DDF','so4_a2_sfnnuc1',
          'bc_a1SFWET','bc_a3SFWET','bc_a4SFWET','bc_c1SFWET','bc_c3SFWET','bc_c4SFWET',
          'dst_a1SFWET','dst_a3SFWET','dst_c1SFWET','dst_c3SFWET',
          'mom_a1SFWET','mom_a2SFWET','mom_a3SFWET','mom_a4SFWET','mom_c1SFWET','mom_c2SFWET','mom_c3SFWET','mom_c4SFWET',
          'ncl_a1SFWET','ncl_a2SFWET','ncl_a3SFWET','ncl_c1SFWET','ncl_c2SFWET','ncl_c3SFWET',
          'pom_a1SFWET','pom_a3SFWET','pom_a4SFWET','pom_c1SFWET','pom_c3SFWET','pom_c4SFWET',
          'so4_a1SFWET','so4_a2SFWET','so4_a3SFWET','so4_a5SFWET','so4_c1SFWET','so4_c2SFWET','so4_c3SFWET','so4_c5SFWET',
          'soa_a1SFWET','soa_a2SFWET','soa_a3SFWET','soa_c1SFWET','soa_c2SFWET','soa_c3SFWET',
          'O3','LHFLX',
          'O3_2DTDA_trop','O3_2DTDB_trop','O3_2DTDD_trop','O3_2DTDE_trop','O3_2DTDI_trop','O3_2DTDL_trop',
          'O3_2DTDN_trop','O3_2DTDO_trop','O3_2DTDS_trop','O3_2DTDU_trop','O3_2DTRE_trop','O3_2DTRI_trop',
          'O3_SRF','NO_2DTDS','NO_TDLgt','NO2_2DTDD','NO2_2DTDS','NO2_TDAcf','CO_SRF','TROPE3D_P','TROP_P',
          'CDNUMC','SFDMS','so4_a1_sfgaex1','so4_a2_sfgaex1','so4_a3_sfgaex1','so4_a5_sfgaex1','soa_a1_sfgaex1',
          'soa_a2_sfgaex1','soa_a3_sfgaex1','GS_soa_a1','GS_soa_a2','GS_soa_a3','AQSO4_H2O2','AQSO4_O3',
          'SFSO2','SO2_CLXF','SO2','DF_SO2','AQ_SO2','GS_SO2','WD_SO2','ABURDENSO4_STR','ABURDENSO4_TRO',
          'ABURDENSO4','ABURDENBC','ABURDENDUST','ABURDENMOM','ABURDENPOM','ABURDENSEASALT',
          'ABURDENSOA','AODSO4_STR','AODSO4_TRO',
          'EXTINCT','AODABS','AODABSBC','CLDICE','CLDLIQ','CLD_CAL_TMPLIQ','CLD_CAL_TMPICE','Mass_bc_srf',
          'Mass_dst_srf','Mass_mom_srf','Mass_ncl_srf','Mass_pom_srf','Mass_so4_srf','Mass_soa_srf','Mass_bc_850',
          'Mass_dst_850','Mass_mom_850','Mass_ncl_850','Mass_pom_850','Mass_so4_850','Mass_soa_850','Mass_bc_500',
          'Mass_dst_500','Mass_mom_500','Mass_ncl_500','Mass_pom_500','Mass_so4_500','Mass_soa_500','Mass_bc_330',
          'Mass_dst_330','Mass_mom_330','Mass_ncl_330','Mass_pom_330','Mass_so4_330','Mass_soa_330','Mass_bc_200',
          'Mass_dst_200','Mass_mom_200','Mass_ncl_200','Mass_pom_200','Mass_so4_200','Mass_soa_200',
	  'O3_2DTDD','O3_2DCIP','O3_2DCIL',  'CO_2DTDS','CO_2DTDD','CO_2DCEP','CO_2DCEL','NO_2DTDD', ! ChemDyg 
          'FLNTC', 'SAODVIS', ! aerosol forcing
          'H2OLNZ'

 fincl2 = 'PS', 'FLUT','PRECT','U200','V200','U850','V850',
          'TCO','SCO','T'
 fincl3 = 'PS', 'PSL','PRECT','TUQ','TVQ','UBOT','VBOT','TREFHT','FLUT','OMEGA500','TBOT','U850','V850','U200','V200','T200','T500','Z700'
 fincl4 = 'PRECT'
 fincl5 = 'O3_SRF'
 fincl6 = 'CO_2DMSD','NO2_2DMSD','NO_2DMSD','O3_2DMSD','O3_2DMSD_trop'

 ! -- chemUCI settings ------------------   
 history_chemdyg_summary = .true.
 history_gaschmbudget_2D = .false.
 history_gaschmbudget_2D_levels = .false.
 history_gaschmbudget_num = 6 !! no impact if  history_gaschmbudget_2D = .false.

 ! -- MAM5 settings ------------------    
 is_output_interactive_volc = .true.                                                                     
                                                                                                               
 ! -- Modified aerosol settings ------   
 n_so4_monolayers_pcage = 3.0D0
 dms_emis_scale = 2.0D0

 ! --- 5x lightning nox tuning ---
 lght_no_prd_factor = 5.D0

 ! --- Retunrd auto-conversion ---
 p3_nc_autocon_expon = -1.10D0


 !--- Autotuning parameters with p3mincdnc set to 5e6 ---
clubb_c1=1.00000000 
clubb_gamma_coef=0.10000000
zmconv_tau=5990.99414347
zmconv_dmpdz=-0.00096924 
zmconv_micro_dcs=0.00010000
nucleate_ice_subgrid=1.26706772
p3_nc_autocon_expon=-0.70000000
p3_qc_accret_expon=1.15330355
zmconv_auto_fac=4.18546052
zmconv_accr_fac=1.50000000
zmconv_ke=0.00000059
cldfrc_dp1=0.06302008
p3_embryonic_rain_size=0.00001500
p3_mincdnc=5000000
 

EOF

cat << EOF >> user_nl_elm
 hist_dov2xy = .true.,.true.
 hist_fincl1 = 'SNOWDP'
 hist_fincl2 = 'H2OSNO', 'FSNO', 'QRUNOFF', 'QSNOMELT', 'FSNO_EFF', 'SNORDSL', 'SNOW', 'FSDS', 'FSR', 'FLDS', 'FIRE', 'FIRA'
 hist_mfilt = 1,365
 hist_nhtfrq = 0,-24
 hist_avgflag_pertape = 'A','A'
 check_finidat_year_consistency = .false.
 check_dynpft_consistency = .false.

 ! --- bi-grid with TOP setting ---
 fsurdat = '\${DIN_LOC_ROOT}/lnd/clm2/surfdata_map/surfdata_0.5x0.5_simyr1850_c200609_with_TOP.nc'
 use_top_solar_rad = .true.
 finidat = '/lcrc/group/e3sm2/ac.xzheng/E3SMv3_dev/20230924.v3alpha04_trigrid.piControl.chrysalis/init/20230901.v3alpha02.tri_grid.ELMSP_TOP.spinup.chrysalis.elm.r.0101-01-01-00000.nc'

EOF

}

# =====================================
# Customize MPAS stream files if needed
# =====================================

patch_mpas_streams() {

echo

}

# =====================================================
# Custom PE layout: custom-N where N is number of nodes
# =====================================================


custom_pelayout() {

if [[ ${run} == custom-* ]];
then
    echo $'\n CUSTOMIZE PROCESSOR CONFIGURATION:'

    # Number of cores per node (machine specific)
    if [ "${MACHINE}" == "chrysalis" ]; then
        ncore=64
	hthrd=2  # hyper-threading
    else
        echo 'ERROR: MACHINE = '${MACHINE}' is not supported for current custom PE layout setting.'
        exit 400
    fi

    # Extract number of nodes
    tmp=($(echo ${run} | tr "-" " "))
    nnodes=${tmp[1]}

    # Applicable to all custom layouts
    pushd ${CASE_SCRIPTS_DIR}
    ./xmlchange NTASKS=1
    ./xmlchange NTHRDS=1
    ./xmlchange ROOTPE=0
    ./xmlchange MAX_MPITASKS_PER_NODE=$ncore
    ./xmlchange MAX_TASKS_PER_NODE=$(( $ncore * $hthrd))

    # Layout-specific customization
    if [ "${nnodes}" == "100" ]; then

       echo Using custom 100 nodes layout

       ### Current defaults for L
      ./xmlchange CPL_NTASKS=5440
      ./xmlchange ATM_NTASKS=5440
      ./xmlchange OCN_NTASKS=960
      ./xmlchange OCN_ROOTPE=5440

      ### Added by Xue for tri-grid
      ./xmlchange LND_NTASKS=576
      ./xmlchange ROF_NTASKS=576
      ./xmlchange ICE_NTASKS=4864
      ./xmlchange LND_ROOTPE=4864
      ./xmlchange ROF_ROOTPE=4864

    else

       echo 'ERRROR: unsupported layout '${PELAYOUT}
       exit 401

    fi

    popd

fi

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
    local repo=E3SM

    echo "Cloning $repo repository branch $BRANCH under $path"
    if [ -d "${path}" ]; then
        echo "ERROR: Directory already exists. Not overwriting"
        exit 20
    fi
    mkdir -p ${path}
    pushd ${path}

    # This will put repository, with all code
    git clone git@github.com:E3SM-Project/${repo}.git .

    # Check out desired branch
    git checkout ${BRANCH}

    # Custom addition
    if [ "${CHERRY}" != "" ]; then
        echo ----- WARNING: adding git cherry-pick -----
        for commit in "${CHERRY[@]}"
        do
            echo ${commit}
            git cherry-pick ${commit}
        done
        echo -------------------------------------------
    fi

    # Bring in all submodule components
    git submodule update --init --recursive

    popd
}

#-----------------------------------------------------
create_newcase() {

    if [ "${do_create_newcase,,}" != "true" ]; then
        echo $'\n----- Skipping create_newcase -----\n'
        return
    fi

    echo $'\n----- Starting create_newcase -----\n'


    if [[ ${PELAYOUT} == custom-* ]];
    then
        layout="M" # temporary placeholder for create_newcase
    else
        layout=${PELAYOUT}

    fi
    ${CODE_ROOT}/cime/scripts/create_newcase \
        --case ${CASE_NAME} \
        --output-root ${CASE_ROOT} \
        --script-root ${CASE_SCRIPTS_DIR} \
        --handle-preexisting-dirs u \
        --compset ${COMPSET} \
        --res ${RESOLUTION} \
        --machine ${MACHINE} \
        --project ${PROJECT} \
        --walltime ${WALLTIME} \
        --pecount ${layout}

    if [ $? != 0 ]; then
      echo $'\nNote: if create_newcase failed because sub-directory already exists:'
      echo $'  * delete old case_script sub-directory'
      echo $'  * or set do_newcase=false\n'
      exit 35
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

    if $NL_MAPS ; then
        echo "Setting nonlinear maps"
        alg=trfvnp2

        # Atm -> srf maps
        a2l=cpl/gridmaps/ne30pg2/map_ne30pg2_to_r05_${alg}.230516.nc
        a2o=cpl/gridmaps/ne30pg2/map_ne30pg2_to_EC30to60E2r2_${alg}.230516.nc
        ./xmlchange ATM2LND_FMAPNAME_NONLINEAR=$a2l
        ./xmlchange ATM2ROF_FMAPNAME_NONLINEAR=$a2l
        ./xmlchange ATM2OCN_FMAPNAME_NONLINEAR=$a2o
    fi

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

    fi

    # Some user_nl settings won't be updated to *_in files under the run directory
    # Call preview_namelists to make sure *_in and user_nl files are consistent.
    echo $'\n----- Preview namelists -----\n'
    ./preview_namelists

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

    elif [ "${MODEL_START_TYPE,,}" == "branch" ] || [ "${MODEL_START_TYPE,,}" == "hybrid" ]; then
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
    patch_mpas_streams

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

