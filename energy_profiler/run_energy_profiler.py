# Profile energy and other measures on the given platform and train a surrogate model

# Author : Shikhar Tuli


import os
import sys

sys.path.append('../txf_design-space/')
sys.path.append('../txf_design-space/flexibert')
sys.path.append('../boshnas/boshnas/')
sys.path.append('./tests/')
sys.path.append('./utils')

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
from matplotlib import pyplot as plt

from embeddings.utils import graph_util, print_util as pu

sys.path.append('../txf_design-space/transformers/src/transformers')
import embedding_util, energy_util

from boshnas import BOSHNAS
from acq import gosh_acq as acq

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from six.moves import cPickle as pickle

from transformers import BertModel
from transformers import RobertaTokenizer, RobertaModel
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_modular_bert import BertModelModular, BertForMaskedLMModular, BertForSequenceClassificationModular

import warnings
warnings.filterwarnings("ignore")


GLUE_TASKS = ['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli']

PREFIX_CHECKPOINT_DIR = "checkpoint"

INIT_SAMPLER = 'Lhs' # Should be in ['Sobol', 'Lhs', 'Halton', Hammersly']
INIT_SAMPLES = 16 # Should be power of 2

RUNS = 3

USE_GPU = True
USE_NCS = False # Either USE_GPU or USE_NCS should be true, when OS is Linux
if USE_NCS: from run_glue_onnx import main as run_glue_onnx

RPI_IP = '10.9.173.6'

CONVERGENCE_UNC_RATIO = 0.05 # Uncertainty w.r.t. the maximum performance value
CONVERGENCE_MSE = 0.003 # MSE of surrogate model on test set
CONVERGENCE_PATIENCE = 5
RANDOM_SAMPLES = 64 # Size of the random sample set to get predictions from surrogate models


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
	print(f'{pu.bcolors.OKBLUE}Training model with hash:{pu.bcolors.ENDC}\n\t{model_hash} \n{pu.bcolors.OKBLUE}and model dictionary:{pu.bcolors.ENDC}\n\t{model_dict}.')

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

	performance_measures = energy_util.get_measures(device, model_path, batch_size, max_seq_length, runs, task, RPI_IP, debug)

	if not debug:
		shutil.rmtree(model_path)

	return performance_measures


def save_dataset(dataset: dict, txf_dataset_file: str):
	"""Save dataset to file
	
	Args:
		dataset (dict): dataset dictionary object
		txf_dataset_file (str): path to save the dataset

	Returns:
		num_evaluated (int): number of points evaluated 
	"""
	json_dataset, num_evaluated = {}, 0
	for key, value in dataset.items():
		json_dataset[key] = {'model_dict': value['model_dict'], 'model_type': value['model_type'], 'embedding': str(value['embedding'])}

		if 'performance' in value.keys():
			num_evaluated += 1
			json_dataset[key]['performance'] = value['performance']

	json.dump(json_dataset, open(txf_dataset_file, 'w+'))
	print(f'{pu.bcolors.OKGREEN}Saved dataset with {len(dataset)} models to: {txf_dataset_file}{pu.bcolors.ENDC}')

	return num_evaluated


def convert_to_tabular(dataset: dict, only_embeddings: bool = False):
	"""Convert dataset to tabular format
	
	Args:
		dataset (dict): dataset dictionary object
		only_embeddings (bool, optional): to return only the embeddings array
	
	Returns:
		X, latency, energy, peak_power (tuple): input embeddings and output performance measures
		or
		X_ds (np.array): input embeddings 
	"""
	X_ds, X, latency, energy, peak_power = [], [], [], [], []
	for model_hash in dataset.keys():
		X_ds.append(dataset[model_hash]['embedding'])
		if 'performance' in dataset[model_hash].keys():
			X.append(dataset[model_hash]['embedding'])
			latency.append(dataset[model_hash]['performance']['latency'])
			energy.append(dataset[model_hash]['performance']['energy'])
			peak_power.append(dataset[model_hash]['performance']['peak_power'])

	X_ds, X, latency, energy, peak_power = np.array(X_ds), np.array(X), np.array(latency), np.array(energy), np.array(peak_power)

	if not only_embeddings:
		return X, latency, energy, peak_power
	else:
		return X_ds


def init_surrogate_models(regressor: str, dataset: dict, design_space: dict, surrogate_models_dir: str, debug: bool = False):
	"""Initialize surrogate models for latency, energy and peak power
	
	Args:
		regressor (str): regressor in ['boshnas', 'gp', 'dt', 'bdt']
		dataset (dict): dataset dictionary object
		design_space (dict): design space loaded from the .yaml file
		surrogate_models_dir (str): directory to store the surrogate models
		debug (bool, optional): to print debug statements

	Returns:
		latency_model, energy_model, peak_power_model (tuple): three surrogate models
	"""
	if regressor == 'boshnas':
		# Get initialization parameters
		embedding_dim = len(dataset[list(dataset.keys())[0]]['embedding'])
		embedding_bounds = embedding_util.get_embedding_bounds(design_space, 'all')
		embedding_bounds = (np.array([bound[0] for bound in embedding_bounds]), np.array([bound[1] for bound in embedding_bounds]))

		latency_model = BOSHNAS(input_dim=embedding_dim,
			bounds=embedding_bounds,
			trust_region=False,
			second_order=True,
			parallel=True if not debug else False,
			model_aleatoric=False,
			save_path=os.path.join(surrogate_models_dir, 'latency'),
			pretrained=False)
		energy_model = BOSHNAS(input_dim=embedding_dim,
			bounds=embedding_bounds,
			trust_region=False,
			second_order=True,
			parallel=True if not debug else False,
			model_aleatoric=False,
			save_path=os.path.join(surrogate_models_dir, 'energy'),
			pretrained=False)
		peak_power_model = BOSHNAS(input_dim=embedding_dim,
			bounds=embedding_bounds,
			trust_region=False,
			second_order=True,
			parallel=True if not debug else False,
			model_aleatoric=False,
			save_path=os.path.join(surrogate_models_dir, 'peak_power'),
			pretrained=False)
	elif regressor == 'gp':
		latency_model = GaussianProcessRegressor(random_state=0)
		energy_model = GaussianProcessRegressor(random_state=0)
		peak_power_model = GaussianProcessRegressor(random_state=0)
	elif regressor  == 'dt':
		latency_model = DecisionTreeRegressor(random_state=0)
		energy_model = DecisionTreeRegressor(random_state=0)
		peak_power_model = DecisionTreeRegressor(random_state=0)
	elif regressor == 'bdt':
		latency_model = AdaBoostRegressor(DecisionTreeRegressor(random_state=0), n_estimators=500)
		energy_model = AdaBoostRegressor(DecisionTreeRegressor(random_state=0), n_estimators=500)
		peak_power_model = AdaBoostRegressor(DecisionTreeRegressor(random_state=0), n_estimators=500)
	else:
		raise NotImplementedError(f'Regressor "{regressor}" has not been implemented')

	return latency_model, energy_model, peak_power_model


def train_surrogate_models(surrogate_models: tuple, X, latency, energy, peak_power, surrogate_models_dir: str):
	"""Trains the surrogate models for latency, energy and peak power
	
	Args:
		surrogate_models (tuple): three surrogate models
		X (np.array): input embeddings
		latency (np.array): latency values
		energy (np.array): energy values
		peak_power (np.array): power values
		surrogate_models_dir (str): directory to store the surrogate models
	"""
	latency_model, energy_model, peak_power_model = surrogate_models
	
	# Train each surrogate model
	if str(type(latency_model)) == "<class 'boshnas.BOSHNAS'>":
		latency_model.train(X, latency)
		energy_model.train(X, energy)
		peak_power_model.train(X, peak_power)
	elif str(type(latency_model)) == "<class 'sklearn.gaussian_process._gpr.GaussianProcessRegressor'>":
		latency_model.fit(X, latency)
		energy_model.fit(X, energy)
		peak_power_model.fit(X, peak_power)

		# Save models
		pickle.dump(latency_model, open(os.path.join(surrogate_models_dir, 'latency.pkl'), 'wb+'), pickle.HIGHEST_PROTOCOL)
		pickle.dump(energy_model, open(os.path.join(surrogate_models_dir, 'energy.pkl'), 'wb+'), pickle.HIGHEST_PROTOCOL)
		pickle.dump(peak_power_model, open(os.path.join(surrogate_models_dir, 'peak_power.pkl'), 'wb+'), pickle.HIGHEST_PROTOCOL)
	elif str(type(latency_model)) == "<class 'sklearn.tree._classes.DecisionTreeRegressor'>":
		latency_model.fit(X, latency)
		energy_model.fit(X, energy)
		peak_power_model.fit(X, peak_power)

		# Save models
		pickle.dump(latency_model, open(os.path.join(surrogate_models_dir, 'latency.pkl'), 'wb+'), pickle.HIGHEST_PROTOCOL)
		pickle.dump(energy_model, open(os.path.join(surrogate_models_dir, 'energy.pkl'), 'wb+'), pickle.HIGHEST_PROTOCOL)
		pickle.dump(peak_power_model, open(os.path.join(surrogate_models_dir, 'peak_power.pkl'), 'wb+'), pickle.HIGHEST_PROTOCOL)
	elif str(type(latency_model)) == "<class 'sklearn.ensemble._weight_boosting.AdaBoostRegressor'>":
		latency_model.fit(X, latency)
		energy_model.fit(X, energy)
		peak_power_model.fit(X, peak_power)

		# Save models
		pickle.dump(latency_model, open(os.path.join(surrogate_models_dir, 'latency.pkl'), 'wb+'), pickle.HIGHEST_PROTOCOL)
		pickle.dump(energy_model, open(os.path.join(surrogate_models_dir, 'energy.pkl'), 'wb+'), pickle.HIGHEST_PROTOCOL)
		pickle.dump(peak_power_model, open(os.path.join(surrogate_models_dir, 'peak_power.pkl'), 'wb+'), pickle.HIGHEST_PROTOCOL)
	else:
		raise RuntimeError(f'Model type "{type(latency_model)}" is not recognized')


def get_predictions(surrogate_models: tuple, X_ds):
	"""Get predictions of latency, energy and peak power from the three surrogate models
	
	Args:
		surrogate_models (tuple): three surrogate models
		X_ds (np.array): input embeddings

	Returns:
		predictions (tuple): tuple of latency, energy and peak power predictions along with maximum total uncertainty and its index
	"""
	latency_model, energy_model, peak_power_model = surrogate_models
	
	# Get predictions from each surrogate model
	if str(type(latency_model)) == "<class 'boshnas.BOSHNAS'>":
		latency_predictions = latency_model.predict(X_ds)
		energy_predictions = energy_model.predict(X_ds)
		peak_power_predictions = peak_power_model.predict(X_ds)

		# Removing aleatoric uncertainties
		latency_predictions = np.array([(pred[0].item(), pred[1][0].item()) for pred in latency_predictions])
		energy_predictions = np.array([(pred[0].item(), pred[1][0].item()) for pred in energy_predictions])
		peak_power_predictions = np.array([(pred[0].item(), pred[1][0].item()) for pred in peak_power_predictions])
	elif str(type(latency_model)) == "<class 'sklearn.gaussian_process._gpr.GaussianProcessRegressor'>":
		latency_predictions = latency_model.predict(X_ds, return_std=True)
		energy_predictions = energy_model.predict(X_ds, return_std=True)
		peak_power_predictions = peak_power_model.predict(X_ds, return_std=True)

		# Converting to a list of tuples
		latency_predictions = np.array([(latency_predictions[0][i], latency_predictions[1][i]) for i in range(len(latency_predictions[0]))])
		energy_predictions = np.array([(energy_predictions[0][i], energy_predictions[1][i]) for i in range(len(energy_predictions[0]))])
		peak_power_predictions = np.array([(peak_power_predictions[0][i], peak_power_predictions[1][i]) for i in range(len(peak_power_predictions[0]))])
	elif str(type(latency_model)) == "<class 'sklearn.tree._classes.DecisionTreeRegressor'>":
		latency_predictions = latency_model.predict(X_ds)
		energy_predictions = energy_model.predict(X_ds)
		peak_power_predictions = peak_power_model.predict(X_ds)

		# Adding zero uncertainties
		latency_predictions = np.array([(latency_predictions[i], 0) for i in range(len(latency_predictions))])
		energy_predictions = np.array([(energy_predictions[i], 0) for i in range(len(energy_predictions))])
		peak_power_predictions = np.array([(peak_power_predictions[i], 0) for i in range(len(peak_power_predictions))])
	elif str(type(latency_model)) == "<class 'sklearn.ensemble._weight_boosting.AdaBoostRegressor'>":
		latency_predictions = latency_model.predict(X_ds)
		energy_predictions = energy_model.predict(X_ds)
		peak_power_predictions = peak_power_model.predict(X_ds)

		dt_predictions = []
		for estimator in latency_model.estimators_:
			pred = estimator.predict(X_ds)
			dt_predictions.append(pred)
		latency_uncertainties = np.std(np.array(dt_predictions), axis=0)

		dt_predictions = []
		for estimator in energy_model.estimators_:
			pred = estimator.predict(X_ds)
			dt_predictions.append(pred)
		energy_uncertainties = np.std(np.array(dt_predictions), axis=0)

		dt_predictions = []
		for estimator in peak_power_model.estimators_:
			pred = estimator.predict(X_ds)
			dt_predictions.append(pred)
		peak_power_uncertainties = np.std(np.array(dt_predictions), axis=0)

		# Adding uncertainties
		latency_predictions = np.array([(latency_predictions[i], latency_uncertainties[i]) for i in range(len(latency_predictions))])
		energy_predictions = np.array([(energy_predictions[i], energy_uncertainties[i]) for i in range(len(energy_predictions))])
		peak_power_predictions = np.array([(peak_power_predictions[i], peak_power_uncertainties[i]) for i in range(len(peak_power_predictions))])
	else:
		raise RuntimeError(f'Model type "{type(latency_model)}" is not recognized')

	max_uncertainty, max_uncertainty_idx = 0, 0
	for i in range(len(latency_predictions)):
		total_uncertainty = latency_predictions[i][1] + energy_predictions[i][1] + peak_power_predictions[i][1]
		if total_uncertainty > max_uncertainty: 
			max_uncertainty = total_uncertainty
			max_uncertainty_idx = i

	return latency_predictions, energy_predictions, peak_power_predictions, max_uncertainty, max_uncertainty_idx


def check_convergence(patience: int, convergence_criterion: str, regressor: str, surrogate_models_dir: str, dataset: dict, design_space: dict, X: 'np.array', latency: 'np.array', energy: 'np.array', peak_power: 'np.array', max_uncertainty: float, debug: bool = False):
    """Check if convergence has been reached
    
    Args:
    	patience (int): current patience value
        convergence_criterion (str): convergence criterion to use, one in ['unc', 'mse']
        regressor (str): regressor in ['boshnas', 'gp', 'dt', 'bdt']
        surrogate_models_dir (str): directory to store the surrogate models
		dataset (dict): dataset dictionary object
		design_space (dict): design space loaded from the .yaml file
        X (np.array): input embeddings
        latency (np.array): latency values
		energy (np.array): energy values
		peak_power (np.array): power values
        max_uncertainty (float): maximum uncertainty in random samples outside of X
        debug (bool, optional): to print debugging statements
    
    Returns:
        convergence_reached, mse, patience (tuple): if convergence has been reached, the mean squared error, and updated patience
    """
    print(f'Current patience: {patience}/{CONVERGENCE_PATIENCE}')
    convergence_reached, mse = False, np.nan

    if not os.path.exists(surrogate_models_dir): os.makedirs(surrogate_models_dir)

    if convergence_criterion == 'unc':
    	if max_uncertainty <= 3 * CONVERGENCE_UNC_RATIO and patience > CONVERGENCE_PATIENCE:
    		convergence_reached = True
    	elif max_uncertainty <= 3 * CONVERGENCE_UNC_RATIO:
    		patience += 1
    elif convergence_criterion == 'mse':
    	len_dataset = X.shape[0]
    	X_train, latency_train, energy_train, peak_power_train = X[:int(0.8 * len_dataset), :], latency[:int(0.8 * len_dataset)], energy[:int(0.8 * len_dataset)], peak_power[:int(0.8 * len_dataset)]
    	X_test, latency_test, energy_test, peak_power_test = X[int(0.8 * len_dataset):, :], latency[int(0.8 * len_dataset):], energy[int(0.8 * len_dataset):], peak_power[int(0.8 * len_dataset):]

    	surrogate_models = init_surrogate_models(regressor, dataset, design_space, surrogate_models_dir, debug)
    	train_surrogate_models(surrogate_models, X_train, latency_train, energy_train, peak_power_train, surrogate_models_dir)

    	latency_predictions, energy_predictions, peak_power_predictions, _, _ = get_predictions(surrogate_models, X_test)
    	mse = mean_squared_error(latency_test, latency_predictions[:, 0]) + mean_squared_error(energy_test, energy_predictions[:, 0]) + mean_squared_error(peak_power_test, peak_power_predictions[:, 0])
    	if mse <= 3 * CONVERGENCE_MSE and patience > CONVERGENCE_PATIENCE:
    		convergence_reached = True
    	elif mse <= 3 * CONVERGENCE_MSE:
    		patience += 1

    shutil.rmtree(surrogate_models_dir)

    return convergence_reached, mse, patience


def main():
	"""Run BOSHCODE to get the best CNN-Accelerator pair in the design space
	"""
	parser = argparse.ArgumentParser(
		description='Input parameters for generation of dataset library',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--regressor',
		metavar='',
		type=str,
		help='regressor method to employ for active learning, one in ["boshnas", "gp", "dt", "bdt"]',
		default='bdt')
	parser.add_argument('--convergence_criterion',
		metavar='',
		type=str,
		help='convergence criterion to use, one in ["unc", "mse"]',
		default='mse')
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
	parser.add_argument('--surrogate_models_dir',
		metavar='',
		type=str,
		help='path to store the BOSHNAS surrogate model',
		default='./dataset/surrogate_models/')
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
		num_evaluated = save_dataset(dataset, args.txf_dataset_file)

	# Run inference initial samples
	for model_hash in dataset.keys():
		if 'performance' in dataset[model_hash].keys(): continue
		dataset[model_hash]['performance'] = \
			worker(args.device, args.models_dir, dataset[model_hash]['model_dict'], model_hash, args.task, args.batch_size, args.max_seq_length, args.runs, args.debug)

	# Save dataset
	num_evaluated = save_dataset(dataset, args.txf_dataset_file)

	# Initialize surrogate models
	surrogate_models = init_surrogate_models(args.regressor, dataset, design_space, args.surrogate_models_dir, args.debug)

	# Get dataset from trained values
	X, latency, energy, peak_power = convert_to_tabular(dataset)
	max_latency, max_energy, max_peak_power = np.amax(latency), np.amax(energy), np.amax(peak_power)

	# Increase maximum values
	max_latency, max_energy, max_peak_power = 2 * max_latency, 2 * max_energy, 2 * max_peak_power
	if args.debug: print(f'Max latency: {max_latency : 0.3f}s/seq. Max energy: {max_energy : 0.3f}J/seq. Max peak power: {max_peak_power : 0.3f}W')

	# Save maximum values for predition later
	if not os.path.exists(args.surrogate_models_dir): os.makedirs(args.surrogate_models_dir)
	json.dump({'max_latency': max_latency, 'max_energy': max_energy, 'max_peak_power': max_peak_power}, open(os.path.join(args.surrogate_models_dir, 'max_values.json'), 'w+'))

	# Train surrogate models on the normalized dataset
	latency, energy, peak_power = latency/max_latency, energy/max_energy, peak_power/max_peak_power
	train_surrogate_models(surrogate_models, X, latency, energy, peak_power, args.surrogate_models_dir)

	# Get random samples of the entire dataset to obtain maximum uncertainty
	random_samples = embedding_util.get_samples(design_space, num_samples=RANDOM_SAMPLES, sampling_method='Random', debug=False)
	X_ds = convert_to_tabular(random_samples, only_embeddings=True)

	# Get predictions from the surrogate models
	latency_predictions, energy_predictions, peak_power_predictions, max_uncertainty, max_uncertainty_idx = get_predictions(surrogate_models, X_ds)

	if args.debug: print(f'Current maximum epistemic uncertainty: {float(max_uncertainty.item()) : 0.3f}, with number of evaluated models: {num_evaluated}')

	max_uncertainties, mse_list, num_evaluated_list = [], [], []

	patience, convergence_reached = 0, False
	while not convergence_reached:
		# Get the most uncertain model
		model_hash = list(random_samples.keys())[max_uncertainty_idx]

		# Add this model to dataset
		dataset[model_hash] = random_samples[model_hash]

		# Run inference on given model
		dataset[model_hash]['performance'] = worker(args.device, args.models_dir, dataset[model_hash]['model_dict'], model_hash, args.task, args.batch_size, args.max_seq_length, args.runs, args.debug)

		# Print prediciton error
		if args.debug: 
			print(f'Predicted values -- Latency: {latency_predictions[max_uncertainty_idx][0] * max_latency : 0.5f}s/seq, Energy: {energy_predictions[max_uncertainty_idx][0] * max_energy : 0.5f}J/seq, Peak power: {peak_power_predictions[max_uncertainty_idx][0] * max_peak_power : 0.5f}W')
			print(f'Evaluated values -- Latency: {dataset[model_hash]["performance"]["latency"]}s/seq, Energy: {dataset[model_hash]["performance"]["energy"]}J/seq, Peak power: {dataset[model_hash]["performance"]["peak_power"]}W')

		# Save dataset
		num_evaluated = save_dataset(dataset, args.txf_dataset_file)

		# Get dataset from trained values
		X, latency, energy, peak_power = convert_to_tabular(dataset)

		# Train surrogate models on the normalized dataset
		latency, energy, peak_power = latency/max_latency, energy/max_energy, peak_power/max_peak_power
		train_surrogate_models(surrogate_models, X, latency, energy, peak_power, args.surrogate_models_dir)

		# Get random samples of the entire dataset to obtain maximum uncertainty
		random_samples = embedding_util.get_samples(design_space, num_samples=RANDOM_SAMPLES, sampling_method='Random', debug=False)
		X_ds = convert_to_tabular(random_samples, only_embeddings=True)

		# Get predictions from the surrogate models
		latency_predictions, energy_predictions, peak_power_predictions, max_uncertainty, max_uncertainty_idx = get_predictions(surrogate_models, X_ds)

		convergence_reached, mse, patience = check_convergence(patience, args.convergence_criterion, args.regressor, './temp', dataset, design_space, X, latency, energy, peak_power, max_uncertainty, args.debug)

		if args.debug: print(f'Current maximum epistemic uncertainty: {float(max_uncertainty.item()) : 0.3f}, and test mean-squared error: {mse : 0.3f}, with number of evaluated models: {num_evaluated}')

		num_evaluated_list.append(num_evaluated); max_uncertainties.append(max_uncertainty); mse_list.append(mse)
		plt.figure()
		plt.plot(num_evaluated_list, max_uncertainties)
		plt.xlabel('Evaluated models')
		plt.ylabel('Norm. max. epistemic uncertainty')
		plt.savefig('./dataset/max_uncertainties.pdf')

		plt.figure()
		plt.plot(num_evaluated_list, mse_list)
		plt.xlabel('Evaluated models')
		plt.ylabel('Test MSE')
		plt.savefig('./dataset/test_mse.pdf')

	print(f'{pu.bcolors.OKGREEN}Convergence criterion reached!{pu.bcolors.ENDC}')


if __name__ == '__main__':
	torch.multiprocessing.set_start_method('spawn')
	main()
