# Create a new environment
conda create -n dipy python=3.11
conda init bash
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

