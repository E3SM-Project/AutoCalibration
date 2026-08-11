# Our old env won't build. Cloning e3sm-unified to start is probably unnecessary.

PACKAGE_ROOT=/global/cfs/cdirs/e3sm/emulate_paper/packages
ENV_ROOT=/global/cfs/cdirs/e3sm/emulate_paper/conda/envs

conda create --prefix $ENV_ROOT/auto_paper python=3.11
conda activate $ENV_ROOT/auto_paper

####### install remaining packages ##################################

mkdir $PACKAGE_ROOT/tesuract
git clone git@github.com:wagmanbe/tesuract.git $PACKAGE_ROOT/tesuract
cd $PACKAGE_ROOT/tesuract
$ENV_ROOT/auto_paper/bin/pip install .

mkdir $PACKAGE_ROOT/clif
git clone git@github.com:sandialabs/clif.git $PACKAGE_ROOT/clif
cd $PACKAGE_ROOT/clif
$ENV_ROOT/auto_paper/bin/pip install .

mkdir $PACKAGE_ROOT/GitPython
git clone git@github.com:sandialabs/clif.git $PACKAGE_ROOT/GitPython
cd $PACKAGE_ROOT/GitPython
$ENV_ROOT/auto_paper/bin/pip install .

$ENV_ROOT/auto_paper/bin/pip install prettytable
$ENV_ROOT/auto_paper/bin/pip install emcee
$ENV_ROOT/auto_paper/bin/pip install tools
$ENV_ROOT/auto_paper/bin/pip install corner
$ENV_ROOT/auto_paper/bin/pip install h5py

 