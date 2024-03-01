sudo apt-get update
sudo apt-get install build-essential

# Assuming that Miniconda is installed in the home directory
source ~/miniconda3/bin/activate
conda init bash
conda create -n dipy python=3.11
conda activate dipy

# Install other packages
cd ~
pip install dask joblib ray matplotlib boto3 numpy
pip install scikit-image
