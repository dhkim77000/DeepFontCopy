#!/bin/bash -f

#=====================================
# MC-GAN
# Train and Test conditional GAN Glyph network
# By Samaneh Azadi
#=====================================


#=====================================
## Set Parameters
#=====================================

DATA=$1
DATASET="/home/dhkim/mc-gan-master/datasets/${DATA}/"
experiment_dir="GlyphNet_pretrain"
MODEL=cGAN
MODEL_G=resnet_6blocks
MODEL_D=n_layers
n_layers_D=1
NORM=batch
IN_NC=26
O_NC=26
GRP=26
PRENET=2_layers
FINESIZE=64
LOADSIZE=64
LAM_A=100
NITER=500
NITERD=100
BATCHSIZE=150
EPOCH=600 #test at which epoch?
CUDA_ID=0

if [ ! -d "/home/dhkim/mc-gan-master/checkpoints/${experiment_dir}" ]; then
	mkdir "/home/dhkim/mc-gan-master/checkpoints/${experiment_dir}"
fi
LOG="/home/dhkim/mc-gan-master/checkpoints/${experiment_dir}/test.txt"
if [ -f $LOG ]; then
	rm $LOG
fi
exec &> >(tee -a "$LOG")


# =======================================
## Test Glyph Network on font dataset
# =======================================
CUDA_VISIBLE_DEVICES=${CUDA_ID} python /home/dhkim/mc-gan-master/test.py --dataroot ${DATASET} --name "${experiment_dir}"\
							 	--model ${MODEL} --which_model_netG ${MODEL_G} --which_model_netD ${MODEL_D} --n_layers_D ${n_layers_D} --which_model_preNet ${PRENET}\
							 	--norm ${NORM} --input_nc ${IN_NC} --output_nc ${O_NC} --grps ${GRP}  --loadSize ${FINESIZE} --fineSize ${LOADSIZE} --display_id 0 --batchSize 1 --conditional\
							 	--which_epoch ${EPOCH} --blanks 0.75 --conv3d --align_data



