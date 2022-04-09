#!/bin/sh

# Script to install required packages in conda from the FlexiBERT repo
# Author : Shikhar Tuli

BOLDYELLOW='\e[1;33m'
ENDC='\e[0m'

if { conda env list | grep ".*txf_design-space*"; } >/dev/null 2>&1
then
	conda activate txf_design-space

	# Install additional libraries
	conda install -c conda-forge treelib
	if [ -d "/home/pi" ]; then
		pip install pi-ina219
	else
		# Install openvivo-dev for Intel NCS2
		conda activate txf_design-space
		pip install openvino-dev[pytorch,onnx]==2021.4.2
		pip install onnxruntime
		pip install transformers[onnx]
		sudo apt install libpython3.9
		conda deactivate
	fi
else
	cd txf_design-space

	if [ "$(uname)" == "Darwin" ]; then
		# Mac OS X platform
		# Conda can be installed from here - https://github.com/conda-forge/miniforge
		echo -e "${BOLDYELLOW}Platform discovered: macOS${ENDC}"

		# Rust needs to be installed
		curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

		# Install environment
		conda create --name txf_design-space python=3.9 pytorch --channel pytorch

		# Install tensorflow dependencies
		conda install -c apple tensorflow-deps

		# Install tensorflow for Apple Silicon
		python -m pip install tensorflow-macos
		python -m pip install tensorflow-metal

		# Install tfds
		conda install -c anaconda tensorflow-datasets

	elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
		# GNU/Linux platform

		# module load anaconda3/2020.11
		if [ -d "/home/pi" ]; then
			# Raspberry Pi platform does not have GPU
			# Conda can be installed from here - https://github.com/conda-forge/miniforge
			echo -e "${BOLDYELLOW}Platform discovered: Linux on Raspberry Pi${ENDC}"

			# Rust needs to be installed
			curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

			# Install environment
			conda create --name txf_design-space python=3.9 pytorch numpy --channel kumatea # https://github.com/KumaTea/pytorch-aarch64
		elif [ -d "/home/nano" ]; then
			# Jetson Nano platform has a Tegra X1 GPU
			# Conda can be installed from here - https://github.com/conda-forge/miniforge
			echo -e "${BOLDYELLOW}Platform discovered: Linux on Nvidia Jetson Nano${ENDC}"

			# Rust needs to be installed
			curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

			# Install environment
			conda create --name txf_design-space python=3.6 

			# Download and install pre-built wheel for Jetson Nano
			conda activate txf_design-space
			wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl -O torch-1.8.0-cp36-cp36m-linux_aarch64.whl -O torch-1.9.0-cp36-cp36m-linux_aarch64.whl # https://pytorch.org/blog/running-pytorch-models-on-jetson-nano/
			sudo apt install python3-pip libopenblas-base libopenmpi-dev 
			pip3 install Cython
			pip3 install numpy torch-1.9.0-cp36-cp36m-linux_aarch64.whl
			export OPENBLAS_CORETYPE=ARMV8
			conda deactivate
		else
			echo -e "${BOLDYELLOW}Platform discovered: GNU Linux${ENDC}"

			# Rust needs to be installed
			curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

			conda create --name txf_design-space python=3.9 pytorch torchvision torchaudio cudatoolkit=11.1 --channel pytorch --channel nvidia

			# Install openvivo-dev for Intel NCS2
			conda activate txf_design-space
			pip install openvino-dev[pytorch,onnx]==2021.4.2
			pip install onnxruntime
			pip install transformers[onnx]
			sudo apt install libpython3.9
			conda deactivate
		fi

	elif [ "$(expr substr $(uname -s) 1 10)" == "MINGW32_NT" ]; then
		# 32 bits Windows NT platform
		echo -e "${BOLDYELLOW}Platform discovered: Windows NT (32-bit)${ENDC}"

		# module load anaconda3/2020.11
		conda create --name txf_design-space python=3.9 pytorch torchvision torchaudio cudatoolkit=11.1 --channel pytorch --channel nvidia

	elif [ "$(expr substr $(uname -s) 1 10)" == "MINGW64_NT" ]; then
		# 64 bits Windows NT platform
		echo -e "${BOLDYELLOW}Platform discovered: Windows NT (64-bit)${ENDC}"

		# module load anaconda3/2020.11
		conda create --name txf_design-space python=3.9 pytorch torchvision torchaudio cudatoolkit=11.1 --channel pytorch --channel nvidia
	fi

	conda activate txf_design-space

	# Already added in current repository
	# git clone https://github.com/huggingface/transformers.git
	cd transformers
	pip install -e .
	pip install torch-dct
	cd ..

	# Install datasets
	git clone https://github.com/huggingface/datasets.git
	cd datasets/
	pip install -e .
	cd ..

	# Add other packages and enabling extentions
	conda install -c conda-forge tqdm ipywidgets matplotlib scikit-optimize
	jupyter nbextension enable --py widgetsnbextension
	conda install -c anaconda scipy cython
	conda install pyyaml
	conda install pandas
	conda install -c plotly plotly

	# Conda prefers pip packages in the end
	pip install grakel
	pip install datasets
	pip install networkx
	pip install tabulate
	pip install optuna

	# Install additional libraries
	conda install -c conda-forge treelib
	if [ -d "/home/pi" ]; then
		pip install pi-ina219
	fi
fi
