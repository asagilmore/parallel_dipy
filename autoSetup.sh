# Assuming that Miniconda is installed in the home directory
source ~/miniconda3/bin/activate
conda init bash
conda create -n dipy python=3.11
conda activate dipy

## install pyafq

cd ~
mkdir src
cd src
git clone https://github.com/asagilmore/dipy.git
cd dipy
pip install -r requirements.txt
pip install .

# Install other packages
cd ~
pip install dask joblib ray matplotlib boto3 numpy

## get code

cd ~
cd parallel_dipy
python3 automaticMeasurment.py
