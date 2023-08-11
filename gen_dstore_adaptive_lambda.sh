#!/bin/bash

# All experiments use probability as memory strength by default unless `--use-perplexity-mem-strength` flag is used

echo "Generating datastore from the training set (w/ dataset shuffling) w/ adaptive lambda"
python eval_lm.py data-bin/wikitext-103 \
    --path checkpoints/checkpoint_best.pt \
    --sample-break-mode none --max-tokens 3072 \
    --softmax-batch 1024 --gen-subset train \
    --context-window 1536 --tokens-per-sample 1536 \
    --dstore-mmap checkpoints/dstore_adaptive_train_shuffled.pt --knn-keytype 'last_ffn_input' \
    --use-adaptive-mem --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --adaptive-mem-log-prob-thresh -1.0 --prune-memory-strength-thresh 0.0001 \
    --memory-decay-factor 0.99 --k 1024 --lmbda 0.25 --probe 32 --shuffle-dataset \
    --use-faiss-index --use-adaptive-lmbda --use-max-weight-lmbda --rbf-beta 0.1 \
    --datastore-update-freq 1 --save-knnlm-dstore --knnlm --fp16 | tee logs/train_shuffled_datastore_adaptive_lambda.log
