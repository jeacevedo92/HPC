#!/bin/bash

#sbatch --job-name=holamundo
#sbatch --gres=gpu:1
#sbatch --output=hola.out

export CUDA_VISIBLE_DEVICES=0
./build/sobel
