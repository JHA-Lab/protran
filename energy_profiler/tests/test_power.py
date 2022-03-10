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


OUTPUT_DIR = './bert_tiny_sst2'
RUNS = 10


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

	elif platform.system() == 'Linux':
		raise RunTimeError(f'Linux is currently unsupported')
	else:
		raise RunTimeError(f'Unsupported OS: {platform.system()}')

	if debug: print(f'CPU Power: {cpu_power} mW \t GPU Power: {gpu_power} mW \t DRAM Power: {dram_power} mW')

	return {'cpu': cpu_power, 'gpu': gpu_power, 'dram': dram_power}


def run_bert_inference(queue, runs: int, output_dir: str):
	"""Run inference of BERT-Tiny on the SST-2 task
	
	Args:
	    queue (mp.Queue): multiprocessing queue
	    runs (int, optional): number of inference runs
	    output_dir (str): directory where the pre-trained model is stored
	
	Returns:
	    dict: evaluation metrics
	"""
	for i in range(runs):
		run_glue(get_training_args(0, output_dir))

	eval_metrics = json.load(open(os.path.join(output_dir, 'eval_results.json'), 'r'))
	queue.put(eval_metrics)


def main():
	# Get mutliprocessing queue
	bert_queue = mp.Queue()

	# Get process
	bert_process = mp.Process(target=run_bert_inference, args=(bert_queue, RUNS, OUTPUT_DIR))

	start_time = time.time()
	power_metrics = []

	# Get power consumption for first 5 iterations
	for i in range(5):
		power_metrics.append({'power_metrics': get_power(debug=True), 'time': time.time() - start_time})

	# Inference starts at power_metrics[4]['time']
	bert_process.start()

	# Get power consumption for 10 more iterations
	for i in range(15):
		power_metrics.append({'power_metrics': get_power(debug=True), 'time': time.time() - start_time})

	eval_metrics = bert_queue.get()
	bert_process.join()

	# Fix timing
	exp_start_time = power_metrics[0]['time']
	for i in range(len(power_metrics)):
		power_metrics[i]['time'] -= exp_start_time

	# Make a plot of all power metrics
	plt.plot([meas['time'] for meas in power_metrics], [meas['power_metrics']['cpu'] for meas in power_metrics], label='CPU Power', color='b')
	plt.plot([meas['time'] for meas in power_metrics], [meas['power_metrics']['gpu'] for meas in power_metrics], label='GPU Power', color='g')
	plt.plot([meas['time'] for meas in power_metrics], [meas['power_metrics']['dram'] for meas in power_metrics], label='DRAM Power', color='r')
	plt.axvline(x=power_metrics[4]['time'], linestyle='--', color='k')
	plt.axvline(x=power_metrics[4]['time']+eval_metrics['eval_runtime']*RUNS, linestyle='--', color='k')
	plt.xlabel('Time (s)')
	plt.ylabel('Power (mW)')
	plt.legend()
	plt.savefig('./results/power_results.pdf')

	json.dump(power_metrics, open('./results/power_metrics.json', 'w+'))

	print(f'Evaluation Accuracy (%): {eval_metrics["eval_accuracy"]*100}. Evaluation Runtime (s): {eval_metrics["eval_runtime"]}')


if __name__ == "__main__":
	main()

