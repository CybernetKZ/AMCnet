#! /bin/bash

python train/infer_classifier.py \
    --input-file data_paths.txt \
    --model-path ./choosen_model/best_model.pt \
    --encoder-model ./cluster_model/encoder-epoch-28-avg-13.onnx \
    --decoder-model ./cluster_model/decoder-epoch-28-avg-13.onnx \
    --joiner-model ./cluster_model/joiner-epoch-28-avg-13.onnx \
    --output-dir inference_results \
    --threshold 0.5
