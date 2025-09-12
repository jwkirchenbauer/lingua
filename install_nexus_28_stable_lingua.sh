# Complete, reproducible script to build and prepare environment

# Exit immediately if a command exits with a non-zero status
set -e

REPO=$(pwd)

# modify the installation path and env name if you want
INSTALLDIR=${WRKSPC}

# on nexus we seem to need this since sometimes the /tmp can get filled up by people's jobs
export TMPDIR="$WRKSPC/.cache/tmp"

ENV_NAME="nexus_28_stable_lingua"

cd ${INSTALLDIR}

# Base the installation on previously installed miniconda.
# Note, this is a manual process currently.
# run this first
# conda_base
# load the runtime modules we will use
# ml cuda/12.9.1

echo "Conda Version:" 
conda env list | grep '*'

# Create conda environment, and print whether it is loaded correctly
# hardcoding the conda of the base miniconda, adjust as necessary
conda create --prefix ${INSTALLDIR}/$ENV_NAME python=3.13.5 --yes -c defaults
source activate ${INSTALLDIR}/$ENV_NAME
echo "Pip Version:" $(which pip)  # should be from the new environment!

# Conda packages:
# might not be anything else we need here
# conda install -c conda-forge conda-pack libstdcxx-ng --yes

######### COMPILE PIP PACKAGES ########################

# pytorch and core reqs
cd "${REPO}"

# 2.8 is now stable
pip install torch==2.8.0+cu129 xformers --index-url https://download.pytorch.org/whl/cu129

pip install --requirement requirements.txt

# extras
pip install hf_transfer wandb

cd ${REPO}
