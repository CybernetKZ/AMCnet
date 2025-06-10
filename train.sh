#! /bin/bash

python train/train_classifier.py \
    --train-labels data/train_labels.txt \
    --val-labels data/val_labels.txt \
    --save-dir exp/models \
    --tensorboard-dir exp/runs \
    --encoder-type fastconformer \
    --encoder-model encoder_model/asr_fastconformer_large_14_universal_kzru_v1_11_01_2025.nemo \
    --batch-size 16 \
    --learning-rate 0.001 \
    --epochs 100 \
    --hidden-dims 1024,512,256 \
    --embedding-dim 512 \
    --sample-rate 16000 \
    --num-workers 4