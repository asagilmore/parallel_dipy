#!/bin/bash
cd ~
sudo apt update
sudo apt install git wget

# Get the absolute path of the home directory
HOME_DIR=$(eval echo ~$USER)

# Install miniConda
mkdir -p $HOME_DIR/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $HOME_DIR/miniconda3/miniconda.sh
bash $HOME_DIR/miniconda3/miniconda.sh -b -u -p $HOME_DIR/miniconda3
rm -rf $HOME_DIR/miniconda3/miniconda.sh

$HOME_DIR/miniconda3/bin/conda init bash