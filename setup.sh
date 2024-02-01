# Assuming that Miniconda is installed in the home directory
source ~/miniconda3/bin/activate
conda init bash
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
sudo apt install python3 python3-dev git curl
conda install -n dipy -c conda-forge mamba
# Add the path to the conda executable to the PATH environment variable
export PATH=~/miniconda3/bin:$PATH

# Verify that conda is in the PATH
which conda

# install ipykernel
conda install -n dipy ipykernel

# Create kernel for dipy env
python -m ipykernel install --prefix=/home/ubuntu/miniconda3/envs/dipy --name=dipy

# Install mamba
conda install -n dipy -c conda-forge mamba

# Run the tljh installation script
curl -L https://tljh.jupyter.org/bootstrap.py | sudo -E python3 - --admin asagilmore