# Test power consumption on different platforms

# Author : Shikhar Tuli


import os
import sys

import platform
import time
import subprocess
import json
import shlex
import multiprocessing as mp
from matplotlib import pyplot as plt

from run_glue import main as run_glue
import torch

if platform.system() == 'Darwin':
	# Running tensorflow version for macOS on arm64
	from run_glue_tf import main as run_glue_tf
	import tensorflow as tf
	

sys.path.append('../../txf_design-space/transformers/src/transformers')


OUTPUT_DIR = './bert_tiny_sst2'
RUNS = 3
USE_GPU = True


def get_training_args(seed, output_dir):
    a = "--seed {} \
    --do_eval \
    --max_seq_length 128 \
    --task_name sst2 \
    --per_device_train_batch_size 32 \
    --learning_rate 1e-4 \
    --model_name_or_path {} \
    --output_dir {} \
        ".format(seed, output_dir, output_dir)
    return shlex.split(a)


def get_power(debug: bool = False):
	"""Get current power consumption
	
	Args:
	    debug (bool, optional): print statements if True

	Raises:
	    RunTimeError: if OS is not supported

	Returns:
	    dict: power metrics in mW
	"""
	cpu_power, gpu_power, dram_power = 0, 0, 0

	if platform.system() == 'Darwin':
		# Get raw output of power metrics
		power_stdout = subprocess.check_output(
			f'sudo powermetrics -n 1',
			shell=True, text=True)

		if power_stdout == 'powermetrics must be invoked as the superuser':
			raise RunTimeError('This script must be run as the superuser')
		else:
			power_stdout = power_stdout.split('\n')
			for line in power_stdout:
				if line.startswith('CPU Power:'):
					cpu_power = int(line.split(' ')[-2])
				elif line.startswith('GPU Power:'):
					gpu_power = int(line.split(' ')[-2])
				elif line.startswith('DRAM Power:'):
					dram_power = int(line.split(' ')[-2])

		if debug: print(f'CPU Power: {cpu_power} mW \t GPU Power: {gpu_power} mW \t DRAM Power: {dram_power} mW')

		return {'cpu': cpu_power, 'gpu': gpu_power, 'dram': dram_power}

	elif platform.system() == 'Linux':
		# Get raw output of nvidia-smi
		power_stdout = subprocess.check_output(
			f'nvidia-smi --query --display=POWER --id=0', # Assuming GPU-id to be 0 for now
			shell=True, text=True)

		power_stdout = power_stdout.split('\n')
		for line in power_stdout:
			if 'Draw' in line.split(): gpu_power = float(line.split()[-2])

		if debug: print(f'GPU Power: {gpu_power} W')

		return {'gpu': gpu_power}

	else:
		raise RunTimeError(f'Unsupported OS: {platform.system()}')


def run_bert_inference(queue, gpu: bool, runs: int, output_dir: str):
	"""Run inference of BERT-Tiny on the SST-2 task
	
	Args:
	    queue (mp.Queue): multiprocessing queue
	    gpu (bool): to run on GPU or not
	    runs (int): number of inference runs
	    output_dir (str): directory where the pre-trained model is stored
	
	Returns:
	    dict: evaluation metrics
	"""
	start_time = time.time()
	for i in range(runs):
		if gpu:
			if platform.system() == 'Darwin':
				run_glue_tf(get_training_args(0, output_dir))
			else:
				os.environ['CUDA_VISIBLE_DEVICES'] = '0'
				run_glue(get_training_args(0, output_dir))
		else:
			if platform.system() == 'Darwin':
				run_glue(get_training_args(0, output_dir))
			else:
				os.environ['CUDA_VISIBLE_DEVICES'] = ''
				run_glue(get_training_args(0, output_dir))

	end_time = time.time()

	if platform.system() == 'Darwin':
		with open(os.path.join(output_dir, 'eval_results.txt')) as file:
			lines = file.readlines()
		eval_metrics = {'eval_loss': float(lines[0].split(' ')[-1]), 'eval_accuracy': float(lines[1].split(' ')[-1]), 'eval_runtime': end_time - start_time}
	else:
		eval_metrics = json.load(open(os.path.join(output_dir, 'eval_results.json'), 'r'))

	queue.put(eval_metrics)


def main():
	# Get mutliprocessing queue
	bert_queue = mp.Queue()

	# Get process
	bert_process = mp.Process(target=run_bert_inference, args=(bert_queue, USE_GPU, RUNS, OUTPUT_DIR))

	start_time = time.time()
	power_metrics = []

	# Get power consumption for first 5 iterations
	for i in range(5):
		power_metrics.append({'power_metrics': get_power(debug=True), 'time': time.time() - start_time})
		if platform.system() == 'Linux': time.sleep(0.1)

	# Inference starts at power_metrics[4]['time']
	bert_process.start()

	if platform.system() == 'Darwin':
		iterations = 25
	else:
		iterations = 100

	# Get power consumption for 10 more iterations
	for i in range(iterations):
		power_metrics.append({'power_metrics': get_power(debug=True), 'time': time.time() - start_time})
		if platform.system() == 'Linux': time.sleep(0.1)

	eval_metrics = bert_queue.get()
	bert_process.join()

	# Fix timing
	exp_start_time = power_metrics[0]['time']
	for i in range(len(power_metrics)):
		power_metrics[i]['time'] -= exp_start_time

	# Make a plot of all power metrics
	if platform.system() == 'Darwin':
		fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
		ax1.plot([meas['time'] for meas in power_metrics], [meas['power_metrics']['cpu']/1000.0 for meas in power_metrics], label='CPU Power', color='b')
		ax2.plot([meas['time'] for meas in power_metrics], [meas['power_metrics']['gpu'] for meas in power_metrics], label='GPU Power', color='g')
		ax3.plot([meas['time'] for meas in power_metrics], [meas['power_metrics']['dram']/1000.0 for meas in power_metrics], label='DRAM Power', color='r')

		ax1.axvline(x=power_metrics[4]['time'], linestyle='--', color='k')
		ax1.axvline(x=power_metrics[4]['time']+eval_metrics['eval_runtime'], linestyle='--', color='k')
		ax2.axvline(x=power_metrics[4]['time'], linestyle='--', color='k')
		ax2.axvline(x=power_metrics[4]['time']+eval_metrics['eval_runtime'], linestyle='--', color='k')
		ax3.axvline(x=power_metrics[4]['time'], linestyle='--', color='k')
		ax3.axvline(x=power_metrics[4]['time']+eval_metrics['eval_runtime'], linestyle='--', color='k')

		ax1.set_xlabel('Time (s)')
		ax2.set_xlabel('Time (s)')
		ax3.set_xlabel('Time (s)')

		ax1.set_ylabel('CPU Power (W)')
		ax2.set_ylabel('GPU Power (mW)')
		ax3.set_ylabel('DRAM Power (W)')

		ax1.set_title(f'Model: BERT-Tiny | Task: SST-2 | Runtime: {eval_metrics["eval_runtime"] : 0.2f}s for {RUNS} runs')
	elif platform.system() == 'Linux':
		fig, ax1 = plt.subplots(1, 1)
		ax1.plot([meas['time'] for meas in power_metrics], [meas['power_metrics']['gpu'] for meas in power_metrics], label='GPU Power', color='g')

		ax1.axvline(x=power_metrics[4]['time'], linestyle='--', color='k')
		ax1.axvline(x=power_metrics[4]['time']+eval_metrics['eval_runtime'], linestyle='--', color='k')

		ax1.set_xlabel('Time (s)')

		ax1.set_ylabel('GPU Power (W)')

		ax1.set_title(f'Model: BERT-Tiny | Task: SST-2 | Runtime: {eval_metrics["eval_runtime"] : 0.2f}s for {RUNS} runs')

	plt.savefig(os.path.join(OUTPUT_DIR, 'power_results.pdf'))

	json.dump(power_metrics, open(os.path.join(OUTPUT_DIR, 'power_metrics.json'), 'w+'))

	print(f'Evaluation Accuracy (%): {eval_metrics["eval_accuracy"]*100}. Evaluation Runtime (s): {eval_metrics["eval_runtime"]}')


if __name__ == "__main__":
	main()

