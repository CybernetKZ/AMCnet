#! /bin/bash

# Test script for Answermachine Audio Classifier
# Supports both Zipformer and FastConformer encoders

python train/test_classifier.py \
    --test-labels data/test_labels.txt \
    --model-path ./classifier_model/best_model.pt \
    --encoder-type wav2vec \
    --encoder-model facebook/wav2vec2-xls-r-300m \
    --output-dir test_results \
    --threshold 0.7

echo "Testing completed. Results in test_results directory."


# Option 1: Test with Zipformer (ONNX) encoder (commented out)
# python train/test_classifier.py \
#     --test-labels data/test_labels.txt \
#     --model-path ./classifier_model/best_model.pt \
#     --encoder-type zipformer \
#     --encoder-model ./encoder_model/encoder-epoch-28-avg-13.onnx \
#     --decoder-model ./encoder_model/decoder-epoch-28-avg-13.onnx \
#     --joiner-model ./encoder_model/joiner-epoch-28-avg-13.onnx \
#     --output-dir test_results \
#     --threshold 0.7

# Option 2: Test with FastConformer encoder
# python train/test_classifier.py \
#     --test-labels data/test_labels.txt \
#     --model-path ./classifier_model/best_model.pt \
#     --encoder-type fastconformer \
#     --encoder-model encoder_model/asr_fastconformer_large_14_universal_kzru_v1_11_01_2025.nemo \
#     --output-dir test_results \
#     --threshold 0.7

# echo "Testing completed. Results in test_results directory."
