CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main.py \
--config_path configs/semantickitti_FastOcc.py \
--log_folder semantickitti_fastocc \
--seed 7240 \
--log_every_n_steps 100 
