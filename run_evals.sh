#!/bin/bash

echo "Generating datastore from the training set"
python eval_lm.py data-bin/wikitext-103 \
    --path checkpoints/checkpoint_best.pt \
    --sample-break-mode none --max-tokens 3072 \
    --softmax-batch 1024 --gen-subset train \
    --context-window 1536 --tokens-per-sample 1536 \
    --dstore-mmap checkpoints/dstore_adaptive_train.pt --knn-keytype 'last_ffn_input' \
    --use-adaptive-mem --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --adaptive-mem-log-prob-thresh -1.0 --prune-memory-strength-thresh 0.1 \
    --memory-decay-factor 0.9 --k 1024 --lmbda 0.25 --probe 32 \
    --datastore-update-freq 1 --save-knnlm-dstore --knnlm --fp16 | tee logs/train_datastore.log

echo "Using a dynamic pretrained (on the training set) datastore on the validation set"
python eval_lm.py data-bin/wikitext-103 \
    --path checkpoints/checkpoint_best.pt \
    --sample-break-mode none --max-tokens 3072 \
    --softmax-batch 1024 --gen-subset valid \
    --context-window 1536 --tokens-per-sample 1536 \
    --dstore-mmap checkpoints/dstore_adaptive_train_val.pt --knn-keytype 'last_ffn_input' \
    --use-adaptive-mem --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --adaptive-mem-log-prob-thresh -1.0 --prune-memory-strength-thresh 0.1 \
    --memory-decay-factor 0.9 --k 1024 --lmbda 0.25 --probe 32 \
    --datastore-update-freq 1 --save-knnlm-dstore --knnlm --fp16 \
    --existing_datastore_path checkpoints/dstore_adaptive_train.pt | tee logs/train_val_datastore.log

echo "Using a dynamic datastore on the validation set"
python eval_lm.py data-bin/wikitext-103 \
    --path checkpoints/checkpoint_best.pt \
    --sample-break-mode none --max-tokens 3072 \
    --softmax-batch 1024 --gen-subset valid \
    --context-window 1536 --tokens-per-sample 1536 \
    --dstore-mmap checkpoints/dstore_adaptive_val.pt --knn-keytype 'last_ffn_input' \
    --use-adaptive-mem --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --adaptive-mem-log-prob-thresh -1.0 --prune-memory-strength-thresh 0.1 \
    --memory-decay-factor 0.9 --k 1024 --lmbda 0.25 --probe 32 \
    --datastore-update-freq 1 --save-knnlm-dstore --knnlm --fp16 | tee logs/val_datastore.log
