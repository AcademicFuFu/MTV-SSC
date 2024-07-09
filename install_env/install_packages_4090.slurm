#!/bin/bash
#SBATCH -J train
#SBATCH -p zjhu

#SBATCH --nodelist=node[34,35,36,37,38,39]
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 4-00:00:00
#SBATCH -o logs.log
#SBATCH --chdir /public/home/wangruoyu2023/envs/CGFormer/4090

echo $(hostname)

srun nvidia-smi
nvcc -V
gcc --version
g++ --version
ninja --version

# 开conda环境
source ~/.bashrc
source ~/miniconda3/bin/activate cgformer-4090

cd /public/home/wangruoyu2023/envs/CGFormer/4090/packages
bash setup.sh
cd ../

sleep 500
###################################################################