#!/bin/bash

set -e

echo "Installing system dependencies..."

sudo apt-get update
sudo apt-get install -y libopenblas-dev libomp-dev

echo "Installing packages with exact versions from ice environment..."


pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118


pip install scipy


pip install numpy==1.26.4 onnxruntime==1.19.2


echo "Fixing invalid distribution..."
pip uninstall -y -umpy || true


echo "Installing k2 from HuggingFace..."
pip install https://huggingface.co/csukuangfj/k2/resolve/main/ubuntu-cuda/k2-1.24.4.dev20240905+cuda11.8.torch2.4.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl


pip install kaldifeat==1.25.4.dev20240905+cuda11.8.torch2.4.1 -f https://csukuangfj.github.io/kaldifeat/cuda.html

echo "Installation complete!" 