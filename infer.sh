#! /bin/bash

python train/infer_classifier.py \
    --input-file data_paths.txt \
    --model-path ./classifier_model/best_model.pt \
    --encoder-model ./encoder_model/encoder-epoch-28-avg-13.onnx \
    --decoder-model ./encoder_model/decoder-epoch-28-avg-13.onnx \
    --joiner-model ./encoder_model/joiner-epoch-28-avg-13.onnx \
    --output-dir inference_results \
    --threshold 0.5
