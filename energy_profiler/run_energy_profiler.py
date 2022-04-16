# Profile energy and other measures on the given platform and train a surrogate model

# Author : Shikhar Tuli


import os
import sys

sys.path.append('../txf_design-space/embeddings')
sys.path.append('../txf_design-space/flexibert')
sys.path.append('../boshnas/boshnas/')
sys.path.append('./tests/')

import yaml
import json
import time
import torch
import shlex
import shutil
import argparse
import subprocess
import collections
import numpy as np
from tqdm import tqdm

from utils import graph_util, print_util as pu

sys.path.append('../../txf_design-space/transformers/src/transformers')
from utils import embedding_util, energy_util

from boshnas import BOSHNAS
from acq import gosh_acq as acq

from transformers import BertModel
from transformers import RobertaTokenizer, RobertaModel
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_modular_bert import BertModelModular, BertForMaskedLMModular, BertForSequenceClassificationModular


GLUE_TASKS = ['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli']

PREFIX_CHECKPOINT_DIR = "checkpoint"

INIT_SAMPLER = 'Lhs' # Should be in ['Sobol', 'Lhs', 'Halton', Hammersly']
INIT_SAMPLES = 4 # Should be power of 2

RUNS = 3

USE_GPU = True
USE_NCS = False # Either USE_GPU or USE_NCS should be true, when OS is Linux
if USE_NCS: from run_glue_onnx import main as run_glue_onnx

RPI_IP = '10.9.173.6'

DELETE_MODELS = False


def worker(device: str, 
	models_dir: str,
	model_dict: dict,
	model_hash: str,
	task: str = 'sst2',
	batch_size: int = 1,
	max_seq_length: int = 128,
	runs: int = 3,
	debug: bool = False):
	"""Worker to run GLUE inference for given model
	
	Args:
		device (str): device in ['cpu', 'gpu', 'npu']
		models_dir (str): path to the models directory
		model_dict (dict): model dictionary of the given model
		model_hash (str): hash of the given model
		task (str): GLUE task to run inference on
		batch_size (int): batch size to be used for running inference
		max_seq_length (int): maximum sequence length for running inference
		runs (int): number of inference runs
		debug (bool, optional): to pring debug statements and save power consumption figures
	
	Returns:
		energy, latency, peak_power (float, float, float): energy, latency and peak power per sequence executed
	"""
	print(f'Training model with hash:\n\t{model_hash} \nand model dictionary:\n\t{model_dict}.')

	assert task in GLUE_TASKS + ['glue'], f'Unsupported task: {task}'
	assert device in ['cpu', 'gpu', 'npu'], f'Unsupported device: {device}'

	# Set model path
	model_path = os.path.join(models_dir, model_hash)

	# Load tokenizer and get model configuration
	tokenizer = RobertaTokenizer.from_pretrained('../txf_design-space/roberta_tokenizer/')
	tokenizer.save_pretrained(model_path)

	config_new = BertConfig(vocab_size = tokenizer.vocab_size)
	config_new.from_model_dict_hetero(model_dict)
	config_new.save_pretrained(model_path)
	
	# Initialize and save given model
	model = BertModelModular(config_new)
	model.save_pretrained(model_path)

	return energy_util.get_measures(device, model_path, batch_size, max_seq_length, runs, task, RPI_IP, debug)


def main():
	"""Run BOSHCODE to get the best CNN-Accelerator pair in the design space
	"""
	parser = argparse.ArgumentParser(
		description='Input parameters for generation of dataset library',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--design_space_file',
		metavar='',
		type=str,
		help='path to the design space configuration file',
		default='./design_space/design_space.yaml')
	parser.add_argument('--txf_dataset_file',
		metavar='',
		type=str,
		help='path to the transformer dataset file',
		default='./dataset/dataset.json')
	parser.add_argument('--models_dir',
		metavar='',
		type=str,
		help='path to the directory where models are stored',
		default='./models/')
	parser.add_argument('--task',
		metavar='',
		type=str,
		help=f'name of GLUE task (or "glue") to train surrogate model for',
		default='sst2')
	parser.add_argument('--batch_size',
		metavar='',
		type=int,
		help=f'batch size to be used for running inference',
		default=1)
	parser.add_argument('--max_seq_length',
		metavar='',
		type=int,
		help='maximum sequence length for the model',
		default=128)
	parser.add_argument('--runs',
		metavar='',
		type=int,
		help='number of runs for measurement of hardware performance',
		default=3)
	parser.add_argument('--device',
		metavar='',
		type=str,
		help=f'device for running inference on',
		default='gpu')
	parser.add_argument('--debug',
		action='store_true',
		help=f'to run debugging statements, save models and power measurement figures',
		default=False)

	args = parser.parse_args()

	random_seed = 0

	# Load design space to run global search on
	design_space = yaml.safe_load(open(args.design_space_file))

	# Load dataset file if previously generated
	if os.path.exists(args.txf_dataset_file):
		dataset = json.load(open(args.txf_dataset_file))
		for key in dataset.keys():
			dataset[key]['embedding'] = eval(dataset[key]['embedding'])
		print(f'{pu.bcolors.OKGREEN}Loaded dataset from: {args.txf_dataset_file}{pu.bcolors.ENDC}')
	else:
		# Generate samples
		dataset = embedding_util.get_samples(design_space, num_samples=INIT_SAMPLES, sampling_method=INIT_SAMPLER, debug=args.debug)

		# Save dataset
		json_dataset = {}
		for key, value in dataset.items():
			json_dataset[key] = {'model_dict': value['model_dict'], 'model_type': value['model_type'], 'embedding': str(value['embedding'])}

		json.dump(json_dataset, open(args.txf_dataset_file, 'w+'))
		print(f'{pu.bcolors.OKGREEN}Saved dataset with {len(dataset)} models to: {args.txf_dataset_file}{pu.bcolors.ENDC}')

	# Runs tests on initial samples
	for model_hash in dataset.keys():
		dataset[model_hash]['performance'] = \
			worker(args.device, args.models_dir, dataset[model_hash]['model_dict'], model_hash, args.task, args.batch_size, args.max_seq_length, args.runs, args.debug)

	print(dataset)


if __name__ == '__main__':
	main()
