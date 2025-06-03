#! /bin/bash

python train/train_classifier.py \
    --train-labels data/train_labels.txt \
    --val-labels data/val_labels.txt \
    --save-dir exp/models \
    --tensorboard-dir exp/runs