#!/bin/sh

python train.py \
        distributed.data_parallel=False \
        common.save_interval=20 \
        common.test_interval=10\
        common.max_epoch=200 \
        common.log_interval=50 \
        datasets.tensor_cut=71_8080  \
        datasets.batch_size=2 \
        datasets.num_workers=9 \
        datasets.train_csv_path=/Users/adees/Code/pytorch-encodec-fork/datasets/diffusion_test.csv \
        datasets.test_csv_path=/Users/adees/Code/pytorch-encodec-fork/datasets/diffusion_test.csv \
        lr_scheduler.warmup_epoch=1 \
        model.sample_rate=24_000 \
        model.causal=False \
        model.norm=time_group_norm \
        model.segment=1.0 \
        model.name=encodec_24khz_reproduce \
        model.channels=1 \
        balancer.weights.l_g=3 \
        balancer.weights.l_feat=3 \
        optimization.lr=3e-4 \
        optimization.disc_lr=1e-4 

# python train_multi_gpu.py \
#         distributed.data_parallel=False \
#         common.save_interval=5000 \
#         common.test_interval=100\
#         common.max_epoch=10000 \
#         common.log_interval=1000 \
#         datasets.tensor_cut=48_000 \
#         datasets.batch_size=16 \
#         datasets.num_workers=9 \
#         datasets.train_csv_path=/Users/adees/Code/pytorch-encodec-fork/datasets/test.csv \
#         datasets.test_csv_path=/Users/adees/Code/pytorch-encodec-fork/datasets/test.csv \
#         lr_scheduler.warmup_epoch=1 \
#         model.sample_rate=48_000 \
#         model.segment=1.0 \
#         model.name=encodec_24khz_reproduce \
#         model.channels=2 \
#         balancer.weights.l_g=4 \
#         balancer.weights.l_feat=4 \
#         optimization.lr=1e-4 \
#         optimization.disc_lr=1e-4
