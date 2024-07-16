#!/bin/bash
#SBATCH -J train
#SBATCH -p zjhu

#SBATCH --nodelist=node[34,35,36,37,38,39]
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 4-00:00:00
#SBATCH -o logs_env_4090.log

echo $(hostname)

srun nvidia-smi
nvcc -V
gcc --version
g++ --version
ninja --version

# 开conda环境
source ~/.bashrc
source ~/miniconda3/bin/activate cgformer

pip uninstall torch_scatter
pip install torch_scatter
cd ../

sleep 500
###################################################################