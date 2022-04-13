# Test power consumption on different platforms

# Author : Shikhar Tuli


import os
import sys

import platform
import time
import subprocess
import json
import shlex
import numpy as np
import multiprocessing as mp
from matplotlib import pyplot as plt

from run_glue import main as run_glue
import torch

if platform.system() == 'Darwin':
	# Running tensorflow version for macOS on arm64
	from run_glue_tf import main as run_glue_tf
	import tensorflow as tf
if os.path.exists('/home/pi'):
	from ina219 import INA219
	

sys.path.append('../../txf_design-space/transformers/src/transformers')


OUTPUT_DIR = './bert_tiny_sst2'
ONNX_DIR = './bert_tiny_onnx'

RPI_IP = '10.9.173.6'

RUNS = 3
USE_GPU = True
USE_NCS = False # Either USE_GPU or USE_NCS should be true, when OS is Linux

if USE_NCS: from run_glue_onnx import main as run_glue_onnx

SHUNT_OHMS = 0.1
INA_ADDRESS = 0x45


def get_training_args(seed, model_name_or_path, output_dir):
    a = "--seed {} \
    --do_eval \
    --max_seq_length 128 \
    --task_name sst2 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --model_name_or_path {} \
    --output_dir {} \
        ".format(seed, model_name_or_path, output_dir)
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

		if os.path.exists('/home/pi/'):
			# To get address of the sensor, use command: sudo i2cdetect -y 1
			# For I2C pinout on Raspberry Pi can be found here: https://pinout.xyz/pinout/i2c
			ina = INA219(shunt_ohms=SHUNT_OHMS, address=INA_ADDRESS)
			ina.configure()
			cpu_power = ina.power()

			if debug: print(f'CPU Power: {cpu_power : 0.02f} mW')

			# TODO: Add support for measuring NPU in Intel NCS2 and CPU/GPU power in Nvidia Jetson Nano
			return {'cpu': cpu_power}
		elif os.path.exists('/home/nano/'):
			# SSH to RPi. We assume keys have been shared already (https://www.thegeekstuff.com/2008/11/3-steps-to-perform-ssh-login-without-password-using-ssh-keygen-ssh-copy-id/)
			power_stdout = subprocess.check_output(
				f'ssh pi@{RPI_IP} ". \'/home/pi/mambaforge/etc/profile.d/conda.sh\'; conda activate txf_design-space; python -c \'from ina219 import INA219; ina = INA219(shunt_ohms={SHUNT_OHMS}, address={INA_ADDRESS}); ina.configure(); print(ina.power())\'"',
				shell=True)
			device_power = float(power_stdout)

			device = 'gpu' if USE_GPU else 'cpu'

			if debug: print(f'{device.upper()}: {device_power : 0.02f} mW')

			return {device: device_power}
		else:
			if USE_GPU:
				# Get raw output of nvidia-smi
				power_stdout = subprocess.check_output(
					f'nvidia-smi --query --display=POWER --id=0', # Assuming GPU-id to be 0 for now
					shell=True, text=True)

				power_stdout = power_stdout.split('\n')
				for line in power_stdout:
					if 'Draw' in line.split(): gpu_power = float(line.split()[-2])

				if debug: print(f'GPU Power: {gpu_power : 0.02f} W')

				return {'gpu': gpu_power}
			elif USE_NCS:
				# SSH to RPi. We assume keys have been shared already (https://www.thegeekstuff.com/2008/11/3-steps-to-perform-ssh-login-without-password-using-ssh-keygen-ssh-copy-id/)
				power_stdout = subprocess.check_output(
					f'ssh pi@{RPI_IP} ". \'/home/pi/mambaforge/etc/profile.d/conda.sh\'; conda activate txf_design-space; python -c \'from ina219 import INA219; ina = INA219(shunt_ohms={SHUNT_OHMS}, address={INA_ADDRESS}); ina.configure(); print(ina.power())\'"',
					shell=True, text=True)
				npu_power = float(power_stdout)

				if debug: print(f'NPU: {npu_power : 0.02f} mW')

				return {'npu': npu_power}

	else:
		raise RunTimeError(f'Unsupported OS: {platform.system()}')


def run_bert_inference(queue, runs: int, output_dir: str):
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
		if USE_GPU:
			if platform.system() == 'Darwin':
				run_glue_tf(get_training_args(0, output_dir, output_dir))
			else:
				os.environ['CUDA_VISIBLE_DEVICES'] = '0'
				run_glue(get_training_args(0, output_dir, output_dir))
		elif USE_NCS:
			if platform.system() != 'Linux':
				raise RunTimeError('Flag variable "USE_NCS" is set to True, but only for Linux platform')
			run_glue_onnx(get_training_args(0, os.path.join(ONNX_DIR, 'model.xml'), ONNX_DIR))
		else:
			if platform.system() == 'Darwin':
				run_glue(get_training_args(0, output_dir, output_dir))
			else:
				os.environ['CUDA_VISIBLE_DEVICES'] = ''
				run_glue(get_training_args(0, output_dir, output_dir))

	end_time = time.time()

	if platform.system() == 'Darwin':
		with open(os.path.join(output_dir, 'eval_results.txt')) as file:
			lines = file.readlines()
		eval_metrics = {'eval_loss': float(lines[0].split(' ')[-1]), 'eval_accuracy': float(lines[1].split(' ')[-1]), 'eval_runtime': end_time - start_time}
	else:
		if USE_NCS:
			eval_metrics = {'eval_loss': np.nan, 'eval_accuracy': np.nan, 'eval_runtime': end_time - start_time}
		else:
			eval_metrics = json.load(open(os.path.join(output_dir, 'eval_results.json'), 'r'))

	queue.put(eval_metrics)


def find_nearest(array, value):
	array = np.asarray(array)
	idx = (np.abs(array - value)).argmin()
	return array[idx], idx


def main():
	# Check if user is root, in case USE_NCS is True
	if USE_NCS:
		stdout = subprocess.check_output('echo $USER', shell=True, text=True).strip()
		if stdout != 'root':
			raise RuntimeError('User should be "root" to use Intel NCS. Use: sudo -E su; source /home/<default_user>/.bashrc; conda activate txf_design-space')
		else:
			stdout = subprocess.check_output('rm -rf /tmp/mvnc.mutex', shell=True, text=True)

	# Get mutliprocessing queue
	bert_queue = mp.Queue()

	# Get process
	bert_process = mp.Process(target=run_bert_inference, args=(bert_queue, RUNS, OUTPUT_DIR))

	start_time = time.time()
	power_metrics = []

	# Get power consumption for first 5 iterations
	for i in range(5):
		power_metrics.append({'power_metrics': get_power(debug=True), 'time': time.time() - start_time})
		if platform.system() == 'Linux': 
			if os.path.exists('/home/pi/'):
				time.sleep(4)
			elif USE_NCS:
				time.sleep(1)
			else:
				time.sleep(0.1)

	# Start inference of BERT-Tiny for {RUNS} runs
	bert_process.start()

	if platform.system() == 'Darwin':
		iterations = 25
	else:
		iterations = 120

	# Initialize evaluation runtime variables
	eval_start_time = 0
	eval_runtime = 0

	# Get power consumption for 10 more iterations
	for i in range(iterations):
		power_metrics.append({'power_metrics': get_power(debug=True), 'time': time.time() - start_time})
		if platform.system() == 'Linux': 
			if os.path.exists('/home/pi/'):
				time.sleep(4)
			elif USE_NCS:
				time.sleep(1)
			else:
				time.sleep(0.1)
		if bert_process.is_alive() and eval_start_time == 0:
			eval_start_time = time.time() - start_time
		if not bert_process.is_alive() and eval_runtime == 0:
			eval_runtime = time.time() - eval_start_time - start_time

	# Get metrics from common queue
	eval_metrics = bert_queue.get()

	# Join process
	bert_process.join()

	# Update evaluation metrics with better runtime estimate
	eval_metrics['eval_runtime'] = eval_runtime

	# Fix timing
	exp_start_time = power_metrics[0]['time']
	for i in range(len(power_metrics)):
		power_metrics[i]['time'] -= exp_start_time

	# Make a plot of all power metrics and get energy
	if platform.system() == 'Darwin':
		# Get energy
		_, eval_start_idx = find_nearest([meas['time'] for meas in power_metrics], eval_start_time)
		_, eval_end_idx = find_nearest([meas['time'] for meas in power_metrics], eval_start_time+eval_metrics['eval_runtime'])
		cpu_energy = np.trapz([meas['power_metrics']['cpu']/1000 for meas in power_metrics][eval_start_idx:eval_end_idx], 
			[meas['time'] for meas in power_metrics][eval_start_idx:eval_end_idx])
		gpu_energy = np.trapz([meas['power_metrics']['gpu']/1000 for meas in power_metrics][eval_start_idx:eval_end_idx], 
			[meas['time'] for meas in power_metrics][eval_start_idx:eval_end_idx])
		dram_energy = np.trapz([meas['power_metrics']['dram']/1000 for meas in power_metrics][eval_start_idx:eval_end_idx], 
			[meas['time'] for meas in power_metrics][eval_start_idx:eval_end_idx])
		total_energy = cpu_energy + gpu_energy + dram_energy

		# Make plot
		fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
		ax1.plot([meas['time'] for meas in power_metrics], [meas['power_metrics']['cpu']/1000.0 for meas in power_metrics], label='CPU Power', color='b')
		ax2.plot([meas['time'] for meas in power_metrics], [meas['power_metrics']['gpu'] for meas in power_metrics], label='GPU Power', color='g')
		ax3.plot([meas['time'] for meas in power_metrics], [meas['power_metrics']['dram']/1000.0 for meas in power_metrics], label='DRAM Power', color='r')

		ax1.axvline(x=eval_start_time, linestyle='--', color='k')
		ax1.axvline(x=eval_start_time+eval_metrics['eval_runtime'], linestyle='--', color='k')
		ax2.axvline(x=eval_start_time, linestyle='--', color='k')
		ax2.axvline(x=eval_start_time+eval_metrics['eval_runtime'], linestyle='--', color='k')
		ax3.axvline(x=eval_start_time, linestyle='--', color='k')
		ax3.axvline(x=eval_start_time+eval_metrics['eval_runtime'], linestyle='--', color='k')

		ax1.set_xlabel('Time (s)')
		ax2.set_xlabel('Time (s)')
		ax3.set_xlabel('Time (s)')

		ax1.set_ylabel('CPU Power (W)')
		ax2.set_ylabel('GPU Power (mW)')
		ax3.set_ylabel('DRAM Power (W)')

		ax1.set_title(f'Model: BERT-Tiny | Task: SST-2 \n Energy: {total_energy/RUNS : 0.2f}J/run | Runtime: {eval_metrics["eval_runtime"] : 0.2f}s for {RUNS} runs')

	elif platform.system() == 'Linux':
		# Get energy
		_, eval_start_idx = find_nearest([meas['time'] for meas in power_metrics], eval_start_time)
		_, eval_end_idx = find_nearest([meas['time'] for meas in power_metrics], eval_start_time+eval_metrics['eval_runtime'])

		if os.path.exists('/home/pi'):
			device = 'cpu'
			energy_mult = 1000
			color = 'b'
		elif USE_NCS:
			device = 'npu'
			energy_mult = 1000
			color = 'tab:orange'
		else:
			device = 'gpu' if USE_GPU else 'cpu'
			energy_mult = 1 if not os.path.exists('/home/nano') else 1000
			color = 'g' if USE_GPU else 'b'

		energy = np.trapz([meas['power_metrics'][device]/energy_mult for meas in power_metrics][eval_start_idx:eval_end_idx], 
			[meas['time'] for meas in power_metrics][eval_start_idx:eval_end_idx])

		# Make plot
		fig, ax1 = plt.subplots(1, 1)
		ax1.plot([meas['time'] for meas in power_metrics], [meas['power_metrics'][device] for meas in power_metrics], label=f'{device.upper()} Power', 
			color=color)

		ax1.axvline(x=eval_start_time, linestyle='--', color='k')
		ax1.axvline(x=eval_start_time+eval_metrics['eval_runtime'], linestyle='--', color='k')

		ax1.set_xlabel('Time (s)')

		ax1.set_ylabel(f'{device.upper()} Power ({"W" if energy_mult == 1 else "mW"})')

		ax1.set_title(f'Model: BERT-Tiny | Task: SST-2 \n Energy: {energy/RUNS : 0.2f}J/run | Runtime: {eval_metrics["eval_runtime"] : 0.2f}s for {RUNS} runs')

	plt.savefig(os.path.join(OUTPUT_DIR, 'power_results.pdf'))

	json.dump(power_metrics, open(os.path.join(OUTPUT_DIR, 'power_metrics.json'), 'w+'))

	print(f'Evaluation Accuracy (%): {eval_metrics["eval_accuracy"]*100}. Evaluation Runtime (s): {eval_metrics["eval_runtime"]}')


if __name__ == "__main__":
	main()

