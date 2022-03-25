# ProTran: Profiling the Energy of Transformers on Embedded Platforms

![Python Version](https://img.shields.io/badge/python-v3.6%20%7C%20v3.7%20%7C%20v3.8-blue)
![Conda](https://img.shields.io/badge/conda%7Cconda--forge-v4.8.3-blue)
![PyTorch](https://img.shields.io/badge/pytorch-v1.8.1-e74a2b)

ProTran is a tool which can be used to generate and evaluate different Transformer architectures on a diverse set of embedded platforms for various natural language processing tasks.
This repository uses the FlexiBERT framework ([jha-lab/txf_design-space](https://github.com/JHA-Lab/txf_design-space)) to obtain the design space of *flexible* and *heterogeneous* Transformer models.

## Table of Contents
- [Environment Setup](#environment-setup)
  - [Clone this repository](#clone-this-repository)
  - [Setup python environment](#setup-python-environment)
- [Replicating results](#replicating-results)

## Environment setup

### Clone this repository

```
git clone --recurse-submodules https://github.com/shikhartuli/protran.git
cd protran
```

### Setup python environment  

The python environment setup is based on conda. The script below creates a new environment named `txf_design-space` or updates an existing environment:
```
source env_step.sh
```
To test the installation, you can run:
```
cd txf_design-space
python check_install.py
cd ..
```
All training scripts use bash and have been implemented using [SLURM](https://slurm.schedmd.com/documentation.html). This will have to be setup before running the experiments.

## Replicating results

Experiments are still being run. Stay tuned!
