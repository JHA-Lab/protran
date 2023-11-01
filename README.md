# ProTran: Profiling the Energy of Transformers on Embedded Platforms

![Python Version](https://img.shields.io/badge/python-v3.6%20%7C%20v3.7%20%7C%20v3.8%20%7C%20v3.9-blue)
![Conda](https://img.shields.io/badge/conda%7Cconda--forge-v4.8.3-blue)
![PyTorch](https://img.shields.io/badge/pytorch-v1.8.1-e74a2b)

ProTran is a tool which can be used to generate and evaluate different Transformer architectures on a diverse set of embedded platforms for various natural language processing tasks.
This repository uses the FlexiBERT framework ([jha-lab/txf_design-space](https://github.com/JHA-Lab/txf_design-space)) to obtain the design space of *flexible* and *heterogeneous* Transformer models.

Supported platforms:
- Linux on x86 CPUs with CUDA GPUs (tested on AMD EPYC Rome CPU, Intel Core i7-8650U CPU and Nvidia A100 GPU).
- Apple M1 and M1-Pro SoC on iPad and MacBook Pro respectively.
- Broadcom BCM2711 SoC on Raspberry Pi 4 Model-B.
- Intel Neural Compute Stick v2.
- Nvidia Tegra X1 SoC on Nvidia Jetson Nano 2GB.

## Table of Contents
- [Environment Setup](#environment-setup)
  - [Clone this repository](#clone-this-repository)
  - [Setup conda](#setup-conda)
  - [Setup python environment](#setup-python-environment)
- [Replicating results](#replicating-results)

## Environment setup

### Clone this repository

```
git clone --recurse-submodules https://github.com/shikhartuli/protran.git
cd protran
```

### Setup conda

For Unix-like platforms, download and install [conda](https://docs.conda.io/en/latest/) through [miniforge](https://github.com/conda-forge/miniforge):

```
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh
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

## Developer

[Shikhar Tuli](https://github.com/shikhartuli). For any questions, comments or suggestions, please reach me at [stuli@princeton.edu](mailto:stuli@princeton.edu).

## Cite this work

Cite our work using the following bitex entry:
```bibtex
@article{tuli2023edgetran,
  title={{EdgeTran}: Device-Aware Co-Search of Transformers for Efficient Inference on Mobile Edge Platforms},
  author={Tuli, Shikhar and Jha, Niraj K},
  journal={IEEE Transactions on Mobile Computing},
  year={2023}
}
```

## License

BSD-3-Clause. 
Copyright (c) 2023, Shikhar Tuli and Jha Lab.
All rights reserved.

See License file for more details.
