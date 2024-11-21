#!/bin/sh
#SBATCH --job-name=vae_encodec
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --partition=LARGE-G2

nvidia-smi

. /home/CAMPUS/d22127229/encodec_env/bin/activate

python train_multi_gpu.py \
        distributed.data_parallel=False \
        common.save_interval=5000 \
        common.test_interval=100\
        common.max_epoch=10000 \
        common.log_interval=1000 \
        datasets.tensor_cut=48_000 \
        datasets.batch_size=16 \
        datasets.num_workers=9 \
        datasets.train_csv_path=/home/CAMPUS/d22127229/code/github/encodec-pytorch/datasets/test_train.csv \
        datasets.test_csv_path=/home/CAMPUS/d22127229/code/github/encodec-pytorch/datasets/test_train.csv \
        lr_scheduler.warmup_epoch=1 \
        model.sample_rate=48_000 \
        model.target_bandwidths="[3., 6., 12., 24.]" \
        model.causal=False \
        model.norm=time_group_norm \
        model.segment=1. \
        model.name=encodec_48khz_reproduce \
        model.channels=2 \
        model.train_discriminator=0.5 \
        balancer.weights.l_g=4 \
        balancer.weights.l_feat=4 \
        optimization.lr=1e-4 \
        optimization.disc_lr=1e-4
