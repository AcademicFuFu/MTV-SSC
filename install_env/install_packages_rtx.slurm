#!/bin/bash
#SBATCH -J train
#SBATCH -p zjhu

#SBATCH --nodelist=node[29,30,31,32,33]
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 4-00:00:00
#SBATCH -o logs_rtx.log
#SBATCH --chdir /public/home/wangruoyu2023/envs/CGFormer/rtx

echo hostname

srun nvidia-smi
nvcc -V
gcc --version
g++ --version
ninja --version

# 开conda环境
source ~/.bashrc
source ~/miniconda3/bin/activate cgformer-rtx

cd /public/home/wangruoyu2023/envs/CGFormer/rtx/packages
bash setup.sh
cd ../

sleep 500
###################################################################