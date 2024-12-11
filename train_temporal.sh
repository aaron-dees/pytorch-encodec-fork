#!/bin/sh

python train_temporal_model.py \
        distributed.data_parallel=False \
        common.save_interval=500 \
        common.test_interval=100\
        common.max_epoch=1000 \
        common.log_interval=100 \
        datasets.tensor_cut=48_000 \
        datasets.batch_size=16 \
        datasets.num_workers=9 \
        datasets.train_csv_path=/Users/adees/Code/pytorch-encodec-fork/datasets/test.csv \
        datasets.test_csv_path=/Users/adees/Code/pytorch-encodec-fork/datasets/test.csv \
        lr_scheduler.warmup_epoch=1 \
        model.sample_rate=48_000 \
        model.target_bandwidths="[3., 6., 12., 24.]" \
        model.causal=False \
        model.norm=time_group_norm \
        model.segment=0.0625 \
        model.name=encodec_48khz_reproduce \
        model.channels=2 \
        model.train_discriminator=0.5 \
        balancer.weights.l_g=4 \
        balancer.weights.l_feat=4 \
        optimization.lr=1e-4 \
        optimization.disc_lr=1e-4
