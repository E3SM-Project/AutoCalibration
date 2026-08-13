# Build the calibration env.

PACKAGE_ROOT=/global/cfs/cdirs/e3sm/emulate_paper/packages
ENV_ROOT=/global/cfs/cdirs/e3sm/emulate_paper/conda/envs
ENV_NAME=auto_paper

mkdir -p "$PACKAGE_ROOT/conda_pkgs"
export CONDA_PKGS_DIRS="$PACKAGE_ROOT/conda_pkgs"

conda create --prefix $ENV_ROOT/$ENV_NAME -c conda-forge \
  python=3.11 \
  numpy=1.26 \
  pandas=1.5.3 \
  scipy=1.13 \
  scikit-learn=1.5.1 \
  xarray \
  dask \
  netcdf4 \
  h5py \
  matplotlib \
  statsmodels \
  pyyaml \
  tqdm \
  gitpython \
  joblib \
  cloudpickle \
  emcee \
  corner \
  cartopy

conda activate $ENV_ROOT/$ENV_NAME

####### install remaining packages ##################################

mkdir $PACKAGE_ROOT
mkdir $PACKAGE_ROOT/tesuract
git clone git@github.com:wagmanbe/tesuract.git $PACKAGE_ROOT/tesuract
cd $PACKAGE_ROOT/tesuract
$ENV_ROOT/$ENV_NAME/bin/pip install .

mkdir $PACKAGE_ROOT/clif
git clone git@github.com:sandialabs/clif.git $PACKAGE_ROOT/clif
cd $PACKAGE_ROOT/clif
$ENV_ROOT/$ENV_NAME/bin/pip install .

mkdir $PACKAGE_ROOT/GitPython
git clone git@github.com:gitpython-developers/GitPython.git $PACKAGE_ROOT/GitPython
cd $PACKAGE_ROOT/GitPython
$ENV_ROOT/$ENV_NAME/bin/pip install .

$ENV_ROOT/$ENV_NAME/bin/pip install prettytable tools

 