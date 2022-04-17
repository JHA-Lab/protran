#!/bin/bash

#SBATCH --job-name=protran		    		# create a short name for your job
#SBATCH --nodes=1                  	    	# node count
#SBATCH --ntasks=1                  	    # total number of tasks across all nodes
#SBATCH --cpus-per-task=8                   # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=256G                          # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1						# number of gpus per node
#SBATCH --time=72:00:00                     # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all                     # send email
#SBATCH --mail-user=stuli@princeton.edu

module purge
module load anaconda3/2020.7
conda activate txf_design-space

python run_energy_profiler.py --batch_size 128 --debug
