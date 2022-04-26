# Utility functions for running inference of FlexiBERT models and obtaining energy, latency and peak power measures

# Author : Shikhar Tuli


import os
import sys

sys.path.append('../tests/')
sys.path.append('../../txf_design-space/transformers/src/transformers')

import json
import time
import torch
import shlex
import shutil
import platform
import subprocess
import numpy as np
from pathlib import Path
import multiprocessing as mp
from matplotlib import pyplot as plt

from run_glue import main as run_glue
try:
    from run_glue_onnx import main as run_glue_onnx
except:
    print('Could not import ONNX library')
if platform.system() == 'Darwin':
    # Running tensorflow version for macOS on arm64
    from run_glue_tf import main as run_glue_tf
    import tensorflow as tf
if os.path.exists('/home/pi'):
    from ina219 import INA219

from convert_graph_to_onnx import convert
from datasets import load_dataset

from transformers import BertModel
from transformers import RobertaTokenizer, RobertaModel
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_modular_bert import BertModelModular, BertForMaskedLMModular, BertForSequenceClassificationModular


SHUNT_OHMS = 0.1
INA_ADDRESS = 0x45

GLUE_TASKS = ['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli']
SUPPORTED_PIPELINES = [
    "feature-extraction",
    "ner",
    "sentiment-analysis",
    "fill-mask",
    "question-answering",
    "text-generation",
    "translation_en_to_fr",
    "translation_en_to_de",
    "translation_en_to_ro",
]


def get_training_args(seed, max_seq_length, batch_size, model_name_or_path, output_dir, task, num_samples):
    if num_samples is None:
        a = "--seed {} \
        --do_eval \
        --max_seq_length {} \
        --task_name {} \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size {} \
        --learning_rate 1e-4 \
        --model_name_or_path {} \
        --output_dir {} \
            ".format(seed, max_seq_length, task, batch_size, model_name_or_path, output_dir)
    else:
        a = "--seed {} \
        --do_eval \
        --max_seq_length {} \
        --task_name {} \
        --max_val_samples {} \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size {} \
        --learning_rate 1e-4 \
        --model_name_or_path {} \
        --output_dir {} \
            ".format(seed, max_seq_length, task, num_samples, batch_size, model_name_or_path, output_dir)
    return shlex.split(a)


def get_power(device: str = 'cpu', rpi_ip: str = None, debug: bool = False):
    """Get current power consumption
    
    Args:
        device (str, optional): device in ['cpu', 'gpu', 'npu']
        rpi_ip (str, optional): IP address of the RPi connected to the INA219 sensor for power measurement
        debug (bool, optional): print statements if True

    Raises:
        RunTimeError: if OS is not supported

    Returns:
        dict: power metrics in mW
    """
    cpu_power, gpu_power, dram_power = 0, 0, 0

    if platform.system() == 'Darwin':
        assert device in ['cpu', 'gpu'], 'Only CPU and GPU supported with macOS'

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
            assert device == 'cpu', 'Only CPU supported with RPi'

            # To get address of the sensor, use command: sudo i2cdetect -y 1
            # For I2C pinout on Raspberry Pi can be found here: https://pinout.xyz/pinout/i2c
            ina = INA219(shunt_ohms=SHUNT_OHMS, address=INA_ADDRESS)
            ina.configure()
            cpu_power = ina.power()

            if debug: print(f'CPU Power: {cpu_power : 0.02f} mW')

            return {'cpu': cpu_power}

        elif os.path.exists('/home/nano/'):
            assert device in ['cpu', 'gpu'], 'Only CPU and GPU supported with Jetson Nano'

            # SSH to RPi. We assume keys have been shared already (https://www.thegeekstuff.com/2008/11/3-steps-to-perform-ssh-login-without-password-using-ssh-keygen-ssh-copy-id/)
            power_stdout = subprocess.check_output(
                f'ssh pi@{rpi_ip} ". \'/home/pi/mambaforge/etc/profile.d/conda.sh\'; conda activate txf_design-space; python -c \'from ina219 import INA219; ina = INA219(shunt_ohms={SHUNT_OHMS}, address={INA_ADDRESS}); ina.configure(); print(ina.power())\'"',
                shell=True)
            device_power = float(power_stdout)

            if debug: print(f'{device.upper()}: {device_power : 0.02f} mW')

            return {device: device_power}

        else:
            if device == 'gpu':
                # Get raw output of nvidia-smi
                power_stdout = subprocess.check_output(
                    f'nvidia-smi --query --display=POWER --id=0', # Assuming GPU-id to be 0 for now
                    shell=True, text=True)

                power_stdout = power_stdout.split('\n')
                for line in power_stdout:
                    if 'Draw' in line.split(): gpu_power = float(line.split()[-2])

                if debug: print(f'GPU Power: {gpu_power : 0.02f} W')

                return {'gpu': gpu_power * 1000}

            elif device == 'npu':
                # SSH to RPi. We assume keys have been shared already (https://www.thegeekstuff.com/2008/11/3-steps-to-perform-ssh-login-without-password-using-ssh-keygen-ssh-copy-id/)
                power_stdout = subprocess.check_output(
                    f'ssh pi@{rpi_ip} ". \'/home/pi/mambaforge/etc/profile.d/conda.sh\'; conda activate txf_design-space; python -c \'from ina219 import INA219; ina = INA219(shunt_ohms={SHUNT_OHMS}, address={INA_ADDRESS}); ina.configure(); print(ina.power())\'"',
                    shell=True, text=True)
                npu_power = float(power_stdout)

                if debug: print(f'NPU: {npu_power : 0.02f} mW')

                return {'npu': npu_power}

    else:
        raise RunTimeError(f'Unsupported OS: {platform.system()}')


def run_bert_inference(queue, device: str, max_seq_length: int, batch_size: int, runs: int, model_path: str, task: str, num_samples: int = None):
    """Run inference of given model on the given GLUE task
    
    Args:
        queue (mp.Queue): multiprocessing queue
        device (str): device in ['cpu', 'gpu', 'npu']
        max_seq_length: maximum sequence length
        batch_size (int): batch size to be used for running inference
        runs (int): number of inference runs
        model_path (str): directory where the model is stored
        task (str): GLUE task to run inference on
        num_samples (int, optional): number of samples in the validation set to run partial inference
    
    Returns:
        dict: evaluation metrics
    """
    start_time = time.time()
    for i in range(runs):
        if device == 'gpu':
            if platform.system() == 'Darwin':
                run_glue_tf(get_training_args(0, max_seq_length, batch_size, model_path, model_path, task, num_samples))
            else:
                # We assume only one GPU is avilable. Else, use: os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                run_glue(get_training_args(0, max_seq_length, batch_size, model_path, model_path, task, num_samples))
        elif device == 'npu':
            if platform.system() != 'Linux':
                raise RunTimeError('device is set to "npu", but only supported for Linux platform')
            run_glue_onnx(get_training_args(0, max_seq_length, batch_size, os.path.join(model_path, 'onnx', 'model.xml'), model_path, task, num_samples))
        else:
            if platform.system() == 'Darwin':
                run_glue(get_training_args(0, max_seq_length, batch_size, model_path, model_path, task, num_samples))
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                run_glue(get_training_args(0, max_seq_length, batch_size, model_path, model_path, task, num_samples))

    end_time = time.time()

    if platform.system() == 'Darwin':
        if device == 'gpu':
            with open(os.path.join(model_path, 'eval_results.txt')) as file:
                lines = file.readlines()
            eval_metrics = {'eval_loss': float(lines[0].split(' ')[-1]), 'eval_accuracy': float(lines[1].split(' ')[-1]), 'eval_runtime': end_time - start_time}
        else:
            eval_metrics = json.load(open(os.path.join(model_path, 'eval_results.json'), 'r'))
    else:
        if device == 'npu':
            eval_metrics = {'eval_loss': np.nan, 'eval_accuracy': np.nan, 'eval_runtime': end_time - start_time}
        else:
            eval_metrics = json.load(open(os.path.join(model_path, 'eval_results.json'), 'r'))

    queue.put(eval_metrics)


def _find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def get_measures(device: str, 
    model_path: str, 
    batch_size: int, 
    max_seq_length: int, 
    runs: int, 
    task: str, 
    num_samples: int = None, 
    rpi_ip: str = None, 
    debug: bool = False):
    """Get hardware performance measures - latency, energy, and peak power consumption per run of inference on the given task
    
    Args:
        device (str): device in ['cpu', 'gpu', 'npu']
        model_path (str): directory where the model is stored
        batch_size (int): batch size to be used for running inference
        max_seq_length (int): maximum sequence length for running inference
        runs (int): number of inference runs
        task (str): GLUE task to run inference on
        num_samples (int, optional): number of samples in the validation set to run partial inference
        rpi_ip (str, optional): IP address of the RPi connected to the INA219 sensor for power measurement
        debug (bool, optional): to pring debug statements and save power consumption figures
    
    Raises:
        RuntimeError: if Intel NCS is to be used, but root user is not selected
    """

    # Check if user is root, in case device == 'npu'
    if device == 'npu':
        stdout = subprocess.check_output('echo $USER', shell=True, text=True).strip()
        if stdout != 'root':
            raise RuntimeError('User should be "root" to use Intel NCS. Use: sudo -E su; source /home/<default_user>/.bashrc; conda activate txf_design-space')
        else:
            stdout = subprocess.check_output('rm -rf /tmp/mvnc.mutex', shell=True, text=True)

        ncs_dir = os.path.join(model_path, 'onnx')
        if os.path.exists(ncs_dir): 
            shutil.rmtree(ncs_dir)
            os.makedirs(ncs_dir)

        # Load tokenizer and model
        tokenizer = RobertaTokenizer.from_pretrained('../txf_design-space/roberta_tokenizer/')
        model = BertForSequenceClassificationModular.from_pretrained(model_path)

        # Convert model to ONNX
        convert(framework='pt', 
            model=model, 
            output=Path(os.path.join(ncs_dir, "model.onnx")), 
            opset=12, 
            tokenizer=tokenizer,
            pipeline_name='sentiment-analysis')

        # Get input dimensions
        input_dim = f'{batch_size} {max_seq_length}'

        # Convert the generated ONNX model to XML format for Intel NCS2
        openvino_stdout = subprocess.check_output(
            f'mo --input_model {os.path.join(ncs_dir, "model.onnx")} --data_type FP16 ' \
            + f' --output_dir {ncs_dir} --input "input_ids[{input_dim}],attention_mask[{input_dim}]"',
            shell=True, text=True)

        print(f'ONNX and XML models saved in the directory: {ncs_dir}')

    # Get mutliprocessing queue
    bert_queue = mp.Queue()

    # Get process
    bert_process = mp.Process(target=run_bert_inference, args=(bert_queue, device, max_seq_length, batch_size, runs, model_path, task, num_samples))

    start_time = time.time()
    power_metrics = []

    # Get power consumption for first 5 iterations
    for i in range(5):
        power_metrics.append({'power_metrics': get_power(device=device, rpi_ip=rpi_ip, debug=debug), 'time': time.time() - start_time})
        if platform.system() == 'Linux': 
            if os.path.exists('/home/pi/'):
                time.sleep(4)
            elif os.path.exists('/home/nano/'):
                time.sleep(0.2)
            elif device == 'npu':
                time.sleep(1)
            else:
                time.sleep(0.2)

    # Start inference of of the given model for 'runs' runs
    bert_process.start()

    if platform.system() == 'Darwin':
        iterations = 400
    else:
        iterations = 1600

    # Initialize evaluation runtime variables
    eval_start_time = 0
    eval_runtime = 0

    # Get power consumption for more iterations
    start_counter, counter = False, 5
    for i in range(iterations):
        if start_counter: counter -= 1 # Inference ended, run for 5 more iterations
        if counter == 0: break
        power_metrics.append({'power_metrics': get_power(device=device, rpi_ip=rpi_ip, debug=True), 'time': time.time() - start_time})
        if platform.system() == 'Linux': 
            if os.path.exists('/home/pi/'):
                time.sleep(4)
            elif os.path.exists('/home/nano/'):
                time.sleep(0.2)
            elif device == 'npu':
                time.sleep(1)
            else:
                time.sleep(0.2)
        if bert_process.is_alive() and eval_start_time == 0:
            eval_start_time = time.time() - start_time
        if not bert_process.is_alive() and eval_runtime == 0:
            eval_runtime = time.time() - eval_start_time - start_time
            start_counter = True

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

    # Find number of sequences in the dataset
    if task != 'glue':
        num_sequences = load_dataset('glue', task, split='validation').num_rows
    else:
        num_sequences = 0
        for task in GLUE_TASKS:
            num_sequences += load_dataset('glue', task, split='validation').num_rows

    if num_samples is not None: num_sequences = num_samples

    # Make a plot of all power metrics and get energy
    if platform.system() == 'Darwin':
        # Get energy
        _, eval_start_idx = _find_nearest([meas['time'] for meas in power_metrics], eval_start_time)
        _, eval_end_idx = _find_nearest([meas['time'] for meas in power_metrics], eval_start_time+eval_metrics['eval_runtime'])
        cpu_energy = np.trapz([meas['power_metrics']['cpu']/1000 for meas in power_metrics][eval_start_idx:eval_end_idx], 
            [meas['time'] for meas in power_metrics][eval_start_idx:eval_end_idx])
        gpu_energy = np.trapz([meas['power_metrics']['gpu']/1000 for meas in power_metrics][eval_start_idx:eval_end_idx], 
            [meas['time'] for meas in power_metrics][eval_start_idx:eval_end_idx])
        dram_energy = np.trapz([meas['power_metrics']['dram']/1000 for meas in power_metrics][eval_start_idx:eval_end_idx], 
            [meas['time'] for meas in power_metrics][eval_start_idx:eval_end_idx])
        energy = cpu_energy + gpu_energy + dram_energy

        # Get peak power in W
        peak_power = max([(meas['power_metrics']['cpu'] + meas['power_metrics']['gpu'] + meas['power_metrics']['dram']) for meas in power_metrics][eval_start_idx:eval_end_idx])/1000 

        # Make plot
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        ax1.plot([meas['time'] for meas in power_metrics], [meas['power_metrics']['cpu'] for meas in power_metrics], label='CPU Power', color='b')
        ax2.plot([meas['time'] for meas in power_metrics], [meas['power_metrics']['gpu'] for meas in power_metrics], label='GPU Power', color='g')
        ax3.plot([meas['time'] for meas in power_metrics], [meas['power_metrics']['dram'] for meas in power_metrics], label='DRAM Power', color='r')

        ax1.axvline(x=eval_start_time, linestyle='--', color='k')
        ax1.axvline(x=eval_start_time+eval_metrics['eval_runtime'], linestyle='--', color='k')
        ax2.axvline(x=eval_start_time, linestyle='--', color='k')
        ax2.axvline(x=eval_start_time+eval_metrics['eval_runtime'], linestyle='--', color='k')
        ax3.axvline(x=eval_start_time, linestyle='--', color='k')
        ax3.axvline(x=eval_start_time+eval_metrics['eval_runtime'], linestyle='--', color='k')

        ax1.set_xlabel('Time (s)')
        ax2.set_xlabel('Time (s)')
        ax3.set_xlabel('Time (s)')

        ax1.set_ylabel('CPU Power (mW)')
        ax2.set_ylabel('GPU Power (mW)')
        ax3.set_ylabel('DRAM Power (mW)')

        ax1.set_title(f'Model: BERT-Tiny | Task: {task} | No. sequences: {num_sequences} \n Energy: {energy/runs : 0.2f}J/run | Runtime: {eval_metrics["eval_runtime"]/runs : 0.2f}s/run')

    elif platform.system() == 'Linux':
        color = 'b'
        if device == 'npu':
            color = 'tab:orange'
        else:
            color = 'g'

        # Get energy
        _, eval_start_idx = _find_nearest([meas['time'] for meas in power_metrics], eval_start_time)
        _, eval_end_idx = _find_nearest([meas['time'] for meas in power_metrics], eval_start_time+eval_metrics['eval_runtime'])
        energy = np.trapz([meas['power_metrics'][device]/1000 for meas in power_metrics][eval_start_idx:eval_end_idx], 
            [meas['time'] for meas in power_metrics][eval_start_idx:eval_end_idx])

        # Get peak power in W
        peak_power = max([meas['power_metrics'][device] for meas in power_metrics][eval_start_idx:eval_end_idx])/1000 

        # Make plot
        fig, ax1 = plt.subplots(1, 1)
        ax1.plot([meas['time'] for meas in power_metrics], [meas['power_metrics'][device] for meas in power_metrics], label=f'{device.upper()} Power', 
            color=color)

        ax1.axvline(x=eval_start_time, linestyle='--', color='k')
        ax1.axvline(x=eval_start_time+eval_metrics['eval_runtime'], linestyle='--', color='k')

        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel(f'{device.upper()} Power (mW)')

        ax1.set_title(f'Model: BERT-Tiny | Task: {task} | No. sequences: {num_sequences} \n Energy: {energy/runs : 0.2f}J/run | Runtime: {eval_metrics["eval_runtime"]/runs : 0.2f}s/run')

    if debug: 
        plt.savefig(os.path.join(model_path, 'power_results.pdf'), bbox_inches='tight')
        print(f'Evaluation Accuracy (%): {eval_metrics["eval_accuracy"]*100}. Evaluation Runtime (s/run): {eval_metrics["eval_runtime"]/runs}')

    json.dump(power_metrics, open(os.path.join(model_path, 'power_metrics.json'), 'w+'))

    protran_results = {'latency': eval_metrics["eval_runtime"]/runs/num_sequences, 'energy': energy/runs/num_sequences, 'peak_power': peak_power}
    json.dump(protran_results, open(os.path.join(model_path, 'protran_results.json'), 'w+'))

    return protran_results


if __name__ == "__main__":
    main()

