#! /bin/bash

python audio_inference.py \
        --input-file data/data_paths.txt \
        --encoder-type wav2vec \
        --encoder-model facebook/wav2vec2-xls-r-300m
