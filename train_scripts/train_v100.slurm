#!/bin/bash                                                                                                                                                                                                                                                                           
#SBATCH -J train
#SBATCH -p zjhu             
#SBATCH --nodelist=node[26,28,22]
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH -t 4-00:00:00
#SBATCH -o logs_v100.log                                                                                                                                                                                             

srun nvidia-smi

# 开conda环境
source ~/miniconda3/bin/activate cgformer-v100
cd $SLURM_SUBMIT_DIR 
export PYTHONPATH=`pwd`:$PYTHONPATH 

read filename <<< $(./train_scripts/utils/filename.sh $1)
config_path=$1
mode=$2
ckpt_path="./logs/$filename/tensorboard/version_0/checkpoints/best.ckpt"
log_folder_eval=""$filename"_eval"

echo "Current working directory: $(pwd)"
echo "Mode: $mode"
echo "Log_folder: $filename"
echo "Config_path: $config_path"
echo "Ckpt_path: $ckpt_path"
echo "Log_folder_eval: $log_folder_eval"

if [ "$mode" = "val" ]; then
    echo "Running validation..."
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python main.py \
    --eval \
    --ckpt_path $ckpt_path \
    --config_path $config_path \
    --log_folder $log_folder_eval \
    --seed 7240 \
    --log_every_n_steps 100 \
    --save_path pred
else
    echo "Running training and validation..."
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python main.py \
    --config_path $config_path \
    --log_folder $filename \
    --seed 7240 \
    --log_every_n_steps 100
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python main.py \
    --eval \
    --ckpt_path $ckpt_path \
    --config_path $config_path \
    --log_folder $log_folder_eval \
    --seed 7240 \
    --log_every_n_steps 100 \
    --save_path pred
fi

###################################################################