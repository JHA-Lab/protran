# Test power consumption on different platforms

# Author : Shikhar Tuli


import os
import sys

sys.path.append('../../txf_design-space/embeddings')
sys.path.append('../../txf_design-space/flexibert')
sys.path.append('../../txf_design-space/transformers/src/transformers')

import numpy as np
import argparse
import subprocess
import shutil
from pathlib import Path
from matplotlib import pyplot as plt
from onnxruntime import InferenceSession
from openvino.inference_engine import IECore

from transformers import BertModel
from transformers import RobertaTokenizer, RobertaModel
from convert_graph_to_onnx import convert
from transformers.models.bert.configuration_bert import BertConfig
from datasets import load_dataset, interleave_datasets, load_from_disk
from transformers.models.bert.modeling_modular_bert import BertModelModular, BertForMaskedLMModular, BertForSequenceClassificationModular


def main():
	"""Convert BERT-Tiny model to ONNX and then XML format for Intel NCS2"""
	parser = argparse.ArgumentParser(
		description='Input parameters for generation of dataset library',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--batch_size',
		metavar='',
		type=int,
		help='batch size',
		default=1)
	parser.add_argument('--max_seq_length',
		metavar='',
		type=int,
		help='maximum sequence length for the model',
		default=128)
	parser.add_argument('--output_dir',
		metavar='',
		type=str,
		help='output directory to store the ONNX and XML files',
		default='./bert_tiny_onnx/')

	args = parser.parse_args()

	# ONNX converter requires a clean directory
	if os.path.exists(args.output_dir) and os.path.isdir(args.output_dir):
		shutil.rmtree(args.output_dir)
	os.makedirs(args.output_dir)

	# Get input dimensions
	input_dim = f'{args.batch_size} {args.max_seq_length}'

	# Load tokenizer
	tokenizer = RobertaTokenizer.from_pretrained('../../txf_design-space/roberta_tokenizer')

	# Define BERT-Tiny model dictionary
	model_dict = {'l': 2, 'o': [['sa_sdp_64']*2]*2, 'h': [128]*2, 'f': [[512]]*2}

	# Load model config
	config = BertConfig(vocab_size = tokenizer.vocab_size)
	config.from_model_dict_hetero(model_dict)

	# Load model
	model = BertForSequenceClassificationModular(config)

	# Convert model to ONNX
	convert(framework='pt', 
	  model=model, 
	  output=Path(os.path.join(args.output_dir, "model.onnx")), 
	  opset=12, 
	  tokenizer=tokenizer,
	  pipeline_name='sentiment-analysis') # sentiment-analysis chosen for sst2 task

	# Convert the generated ONNX model to XML format for Intel NCS2
	openvino_stdout = subprocess.check_output(
		f'mo --input_model {os.path.join(args.output_dir, "model.onnx")} --data_type FP16 ' \
		+ f' --output_dir {args.output_dir} --input "input_ids[{input_dim}],attention_mask[{input_dim}]"',
		shell=True, text=True)

	print(f'ONNX and XML models saved in the directory: {args.output_dir}')


if __name__ == "__main__":
	main()

