#!/bin/bash

job="adapt_in_wiki103"
srun -p V100-32GB -N1 --ntasks-per-node=4 --gpus-per-task=1 --cpus-per-gpu=6 --mem=195G \
    --kill-on-bad-exit --job-name ${job} --nice=0 --time 5-00:00:00  --gpu-bind=none \
    --container-mounts=/netscratch:/netscratch,/ds:/ds,/home/siddiqui:/home/siddiqui --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.04-py3.sqsh \
    --container-workdir=`pwd` --container-mount-home --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5"${EXPORTS} \
    python train.py --task language_modeling \
        data-bin/wikitext-103 \
        --save-dir checkpoints_wiki103/ \
        --arch transformer_lm_wiki103 \
        --max-update 286000 --max-lr 1.0 --t-mult 2 --lr-period-updates 270000 --lr-scheduler cosine --lr-shrink 0.75 \
        --warmup-updates 16000 --warmup-init-lr 1e-07 --min-lr 1e-09 --optimizer nag --lr 0.0001 --clip-norm 0.1 \
        --criterion adaptive_loss --max-tokens 3072 --update-freq 3 --tokens-per-sample 3072 --seed 1 --fp16 \
        --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d \
        --distributed-no-spawn > ./logs/${job}.log 2>&1 &
