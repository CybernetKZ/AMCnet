#! /bin/bash

python train/test_classifier.py \
    --test-labels dataset/test_labels.txt \
    --model-path ./classifier_model/best_model.pt \
    --encoder-model ./encoder_model/encoder-epoch-28-avg-13.onnx \
    --decoder-model ./encoder_model/decoder-epoch-28-avg-13.onnx \
    --joiner-model ./encoder_model/joiner-epoch-28-avg-13.onnx \
    --output-dir test_results