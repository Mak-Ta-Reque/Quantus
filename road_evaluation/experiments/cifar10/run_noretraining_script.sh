#!/bin/bash
# Call this script from the experiments/cifar10 folder as cwd.
export PYTHONPATH=../../
python NoRetrainingMethod.py \
		--data_path=/workspaces/Quantus/road_evaluation-main/data/ig\
		--expl_path=/workspaces/Quantus/road_evaluation-main/data/ig\
		--params_file='noretrain_params.json' \
		--model_path='../../data/cifar_8014.pth'\
