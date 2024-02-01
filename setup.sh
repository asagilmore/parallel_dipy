#!/bin/bash
sudo apt update
sudo apt install git curl

# Install miniConda
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

~/miniconda3/bin/conda init bash

# Source the bashrc file to apply the changes immediately
source ~/.bashrc

# Create a new environment
conda create -n dipy python=3.11
conda activate dipy

# Install the required packages
cd ~
mkdir src
cd src
git clone https://github.com/asagilmore/dipy.git
cd dipy
pip install -r requirements.txt
pip install .

# Install other packages
cd ~
pip install dask joblib ray matplotlib

# setup Jupyter hub
curl -L https://tljh.jupyter.org/bootstrap.py | sudo -E python3 - --admin asagilmore

