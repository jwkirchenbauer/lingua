# Complete, reproducible script to build and prepare environment
CURR_REPO=$(pwd)

# modify the installation path and env name if you want
INSTALLDIR=${WRKSPC}
ENV_NAME="tuolumne_conda_25_621_lingua"

cd ${INSTALLDIR}

# Base the installation on previously installed miniconda.
# Note, this is a manual process currently.

echo "Conda Version:" 
conda env list | grep '*'

# Create conda environment, and print whether it is loaded correctly
conda create --prefix ${INSTALLDIR}/$ENV_NAME python=3.12 --yes -c defaults
source activate ${INSTALLDIR}/$ENV_NAME
echo "Pip Version:" $(which pip)  # should be from the new environment!

# Conda packages:
conda install -c conda-forge conda-pack libstdcxx-ng --yes

# Load modules
rocm_version=6.2.1

module load rocm/$rocm_version
module load gcc-native/12.2

######### COMPILE PIP PACKAGES ########################

# pytorch and core reqs
MAX_JOBS=48 PYTORCH_ROCM_ARCH='gfx942' GPU_ARCHS='gfx942' pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/rocm6.2
pip install ninja packaging numpy

cd ${INSTALLDIR}
git clone https://github.com/ROCm/xformers.git xformers_25_621_lingua
cd xformers_25_621_lingua
git submodule update --init --recursive
MAX_JOBS=48 PYTORCH_ROCM_ARCH=gfx942 python setup.py install
cd ${INSTALLDIR}
rm -rf xformers_25_621_lingua

cd "${CURR_REPO}"
MAX_JOBS=48 PYTORCH_ROCM_ARCH='gfx942' GPU_ARCHS='gfx942' pip install -r requirements.txt
cd ${INSTALLDIR}

# amdsmi
cp -R /opt/rocm-${rocm_version}/share/amd_smi/ $WRKSPC/amd_smi_${rocm_version}
cd $WRKSPC/amd_smi_${rocm_version}
pip install .
cd ${INSTALLDIR}