#! /bin/bash

python train/test_classifier.py \
    --test-labels dataset/test_labels.txt \
    --model-path ./choosen_model/best_model.pt \
    --encoder-model ./cluster_model/encoder-epoch-28-avg-13.onnx \
    --decoder-model ./cluster_model/decoder-epoch-28-avg-13.onnx \
    --joiner-model ./cluster_model/joiner-epoch-28-avg-13.onnx \
    --output-dir test_results