
<div align="center">

# Dacon baseline template

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## Description
Dacon baseline code

## How to run
Install dependencies
```yaml
# clone project
git clone https://github.com/Yoontae6719/Dacon_baseline
cd your-repo-name


# install requirements
pip install -r requirements.txt
```

Train model with default configuration
```yaml
# default
python run.py

# train on GPU
python trainer.py --exp_name dacon_data --conf_file_file ./conf/dacon_data.yaml -seed 20205289


