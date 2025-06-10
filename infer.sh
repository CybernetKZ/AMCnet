#! /bin/bash

python audio_inference.py \
        --input-file data/data_paths.txt \
        --encoder-type fastconformer \
        --encoder-model encoder_model/asr_fastconformer_large_14_universal_kzru_v1_11_01_2025.nemo
