#!/bin/bash

echo "Using a dynamic pretrained (on the training set w/o shuffling) datastore on the validation set"
python eval_lm.py data-bin/wikitext-103 \
    --path checkpoints/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 2560 --tokens-per-sample 2560 \
    --context-window 2560 --softmax-batch 1024 --gen-subset valid \
    --dstore-mmap checkpoints/dstore_adaptive_train_val.pt --knn-keytype 'last_ffn_input' \
    --use-adaptive-mem --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --adaptive-mem-log-prob-thresh -1.0 --prune-memory-strength-thresh 0.001 \
    --memory-decay-factor 0.9 --k 1024 --lmbda 0.25 --probe 32 \
    --datastore-update-freq 1 --save-knnlm-dstore --knnlm --fp16 --freeze-loaded-memories \
    --existing-datastore-path checkpoints/dstore_adaptive_train.pt > logs/train_val_datastore.log 2>&1

echo "Using a dynamic pretrained (on the training set w/o shuffling) datastore on the validation set with adaptive lambda"
python eval_lm.py data-bin/wikitext-103 \
    --path checkpoints/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 2560 --tokens-per-sample 2560 \
    --context-window 2560 --softmax-batch 1024 --gen-subset valid \
    --dstore-mmap checkpoints/dstore_adaptive_train_val.pt --knn-keytype 'last_ffn_input' \
    --use-adaptive-mem --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --adaptive-mem-log-prob-thresh -1.0 --prune-memory-strength-thresh 0.001 \
    --memory-decay-factor 0.9 --k 1024 --lmbda 0.25 --probe 32 \
    --datastore-update-freq 1 --save-knnlm-dstore --knnlm --fp16 \
    --freeze-loaded-memories --use-adaptive-lmbda \
    --existing-datastore-path checkpoints/dstore_adaptive_train.pt > logs/train_val_datastore_adaptive_lmbda.log 2>&1

echo "Using a dynamic pretrained (on the training set w/ shuffling) datastore on the validation set"
python eval_lm.py data-bin/wikitext-103 \
    --path checkpoints/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 2560 --tokens-per-sample 2560 \
    --context-window 2560 --softmax-batch 1024 --gen-subset valid \
    --dstore-mmap checkpoints/dstore_adaptive_train_shuffled_val.pt --knn-keytype 'last_ffn_input' \
    --use-adaptive-mem --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --adaptive-mem-log-prob-thresh -1.0 --prune-memory-strength-thresh 0.001 \
    --memory-decay-factor 0.9 --k 1024 --lmbda 0.25 --probe 32 \
    --datastore-update-freq 1 --save-knnlm-dstore --knnlm --fp16 --freeze-loaded-memories \
    --existing-datastore-path checkpoints/dstore_adaptive_train_shuffled.pt > logs/train_shuffled_val_datastore.log 2>&1

echo "Using a dynamic pretrained (on the training set w/ shuffling) datastore on the validation set with adaptive lambda"
python eval_lm.py data-bin/wikitext-103 \
    --path checkpoints/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 2560 --tokens-per-sample 2560 \
    --context-window 2560 --softmax-batch 1024 --gen-subset valid \
    --dstore-mmap checkpoints/dstore_adaptive_train_shuffled_val.pt --knn-keytype 'last_ffn_input' \
    --use-adaptive-mem --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --adaptive-mem-log-prob-thresh -1.0 --prune-memory-strength-thresh 0.001 \
    --memory-decay-factor 0.9 --k 1024 --lmbda 0.25 --probe 32 \
    --datastore-update-freq 1 --save-knnlm-dstore --knnlm --fp16 \
    --freeze-loaded-memories --use-adaptive-lmbda \
    --existing-datastore-path checkpoints/dstore_adaptive_train_shuffled.pt > logs/train_shuffled_val_datastore_adaptive_lmbda.log 2>&1

echo "Using a dynamic datastore on the validation set"
python eval_lm.py data-bin/wikitext-103 \
    --path checkpoints/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 2560 --tokens-per-sample 2560 \
    --context-window 2560 --softmax-batch 1024 --gen-subset valid \
    --dstore-mmap checkpoints/dstore_adaptive_val.pt --knn-keytype 'last_ffn_input' \
    --use-adaptive-mem --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --adaptive-mem-log-prob-thresh -1.0 --prune-memory-strength-thresh 0.001 \
    --memory-decay-factor 0.9 --k 1024 --lmbda 0.25 --probe 32 --freeze-loaded-memories \
    --datastore-update-freq 1 --save-knnlm-dstore --knnlm --fp16 > logs/val_datastore.log 2>&1

echo "Using a dynamic datastore on the validation set with adaptive lambda"
python eval_lm.py data-bin/wikitext-103 \
    --path checkpoints/checkpoint_best.pt \
    --sample-break-mode complete --max-tokens 2560 --tokens-per-sample 2560 \
    --context-window 2560 --softmax-batch 1024 --gen-subset valid \
    --dstore-mmap checkpoints/dstore_adaptive_val.pt --knn-keytype 'last_ffn_input' \
    --use-adaptive-mem --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --adaptive-mem-log-prob-thresh -1.0 --prune-memory-strength-thresh 0.001 \
    --memory-decay-factor 0.9 --k 1024 --lmbda 0.25 --probe 32 \
    --freeze-loaded-memories --use-adaptive-lmbda \
    --datastore-update-freq 1 --save-knnlm-dstore --knnlm --fp16 > logs/val_datastore_adaptive_lmbda.log 2>&1
