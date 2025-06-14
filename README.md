# Answermachine Audio Classifier

This project implements an audio classification system that can identify whether an audio contains an answermachine message or not. It extracts audio embeddings from pre-trained encoder models and trains a binary classifier with 2 hidden layers.

## Features

- Audio data loading and preprocessing
- Feature extraction using Fbank features
- Support for multiple encoder types:
  - **Zipformer models**: ONNX-based encoder models
  - **FastConformer models**: NeMo-based encoder models from NVIDIA
  - **Wav2Vec models**: HuggingFace-based encoder models
- Binary classifier with configurable hidden layers
- Training and validation pipeline
- Metrics for binary classification (accuracy, precision, recall, F1, AUC)
- Support for class weights to handle imbalanced data
- Batch inference processing with memory management
- Automatic result file management and progress tracking

## Encoder Model Support

### Zipformer (ONNX Models)
- Uses ONNX Runtime for inference
- Requires encoder, decoder, and joiner model files
- Optimized for CPU/GPU execution
- Features extracted using Kaldi Fbank

### FastConformer (NeMo Models)
- Uses NVIDIA NeMo framework
- Supports both pre-trained models and custom .nemo files
- GPU-accelerated inference
- Direct audio processing with built-in preprocessing

### Wav2Vec (HuggingFace Models)
- Uses HuggingFace Transformers library
- Supports various wav2vec2 models (e.g., facebook/wav2vec2-xls-r-300m)
- GPU-accelerated inference
- Direct audio processing with built-in preprocessing
- No additional model files required - downloads automatically

## Requirements

- Python 3.10.16
- PyTorch
- torchaudio
- kaldifeat
- onnxruntime
- k2 (for some model components)
- tqdm
- scikit-learn
- **nemo-toolkit** (for FastConformer support)

## installation
1. create env
```bash
conda create -n classifier python==3.10.16
```
2. activate the env
```bash
conda activate classifier
```
3. setup env
```bash
bash setup_env.sh
```
4. install requirements
```bash
pip install -r requirements.txt
```

## Usage

### Prepare Data

The project includes a `data_preparation.py` script that automates the data preparation process.

#### 1. Create Dataset CSV

First, create a CSV file (`data/dataset.csv`) with the following columns:
- `audio_filepath`: Full path to the audio file
- `label`: Binary label (0 = not answermachine, 1 = answermachine)

Example CSV format:
```csv
audio_filepath,label
/path/to/audio1.wav,0
/path/to/audio2.wav,1
/path/to/audio3.wav,0
```

#### 2. Run Data Preparation

Execute the data preparation script:
```bash
python data_preparation.py
```

This script will:
- **Validate audio files**: Check that all audio files exist and are readable
- **Remove invalid files**: Skip corrupted or missing audio files
- **Split dataset**: Automatically create train/validation/test splits (90%/5%/5%)
- **Generate label files**: Create properly formatted label files for training

#### 3. Output Files

The script generates three label files in the `data/` directory:
- `data/train_labels.txt`: Training set labels
- `data/val_labels.txt`: Validation set labels  
- `data/test_labels.txt`: Test set labels

Each file contains one line per audio file in the format:
```
/path/to/audio1.wav 0
/path/to/audio2.wav 1
/path/to/audio3.wav 0
```

#### 4. Customization Options

You can modify the split ratios in `data_preparation.py`:
```python
prepare_training_data(df, 
                     train_ratio=0.9,   # 90% for training
                     val_ratio=0.05,    # 5% for validation  
                     test_ratio=0.05,   # 5% for testing
                     random_state=42)   # For reproducibility
```

#### 5. Data Validation Features

The preparation script includes:
- **Audio file validation**: Ensures files exist and are readable
- **Label validation**: Confirms labels are binary (0 or 1)
- **Statistics reporting**: Shows dataset distribution and split information
- **Error handling**: Reports invalid files and skips them gracefully

### Training

#### 1. Easy Way - Zipformer
Run 
```bash
bash train.sh
``` 

#### 2. Easy Way - FastConformer
```bash
python train/train_classifier.py \
  --train-labels data/train_labels.txt \
  --val-labels data/val_labels.txt \
  --encoder-type fastconformer \
  --encoder-model nvidia/stt_kk_ru_fastconformer_hybrid_large \
  --batch-size 16 \
  --epochs 10 \
  --learning-rate 0.001 \
  --hidden-dims 1024,512,256 \
  --embedding-dim 512 \
  --save-dir models \
  --sample-rate 16000
```

#### 3. Manual Training - Zipformer
```bash
python train/train_classifier.py \
  --train-labels data/train_labels.txt \
  --val-labels data/val_labels.txt \
  --encoder-type zipformer \
  --encoder-model=/path/to/encoder.onnx \
  --decoder-model=/path/to/decoder.onnx \
  --joiner-model=/path/to/joiner.onnx \
  --batch-size=32 \
  --epochs=10 \
  --learning-rate=0.001 \
  --hidden-dims=256,128 \
  --embedding-dim=512 \
  --save-dir=models \
  --sample-rate=16000
```

#### 4. Manual Training - FastConformer
```bash
python train/train_classifier.py \
  --train-labels data/train_labels.txt \
  --val-labels data/val_labels.txt \
  --encoder-type fastconformer \
  --encoder-model nvidia/stt_kk_ru_fastconformer_hybrid_large \
  --batch-size=16 \
  --epochs=10 \
  --learning-rate=0.001 \
  --hidden-dims=1024,512,256 \
  --embedding-dim=512 \
  --save-dir=models \
  --sample-rate=16000
```

### Training with Custom FastConformer Models

#### Using Pre-trained NeMo Models
```bash
python train/train_classifier.py \
  --train-labels data/train_labels.txt \
  --val-labels data/val_labels.txt \
  --encoder-type fastconformer \
  --encoder-model nvidia/stt_en_fastconformer_hybrid_large_streaming \
  --batch-size=16
```

#### Using Local .nemo Files
```bash
python train/train_classifier.py \
  --train-labels data/train_labels.txt \
  --val-labels data/val_labels.txt \
  --encoder-type fastconformer \
  --encoder-model path/to/your/model.nemo \
  --batch-size=16
```

### Training with Class Weights

If your dataset is imbalanced (e.g., more non-answermachine samples than answermachine), you can use class weights with both encoder types:

```bash
# With FastConformer
python train/train_classifier.py \
  --train-labels data/train_labels.txt \
  --val-labels data/val_labels.txt \
  --encoder-type fastconformer \
  --encoder-model nvidia/stt_kk_ru_fastconformer_hybrid_large \
  --class-weights=1.0,4.0  # weight for non-answermachine, weight for answermachine

# With Zipformer  
python train/train_classifier.py \
  --train-labels data/train_labels.txt \
  --val-labels data/val_labels.txt \
  --encoder-type zipformer \
  --encoder-model=/path/to/encoder.onnx \
  --decoder-model=/path/to/decoder.onnx \
  --joiner-model=/path/to/joiner.onnx \
  --class-weights=1.0,4.0
```

### Training with Wav2Vec Models

#### Using Pre-trained HuggingFace Models
```bash
python train/train_classifier.py \
  --train-labels data/train_labels.txt \
  --val-labels data/val_labels.txt \
  --encoder-type wav2vec \
  --encoder-model facebook/wav2vec2-xls-r-300m \
  --batch-size=16 \
  --epochs=10 \
  --learning-rate=0.001 \
  --hidden-dims=1024,512,256 \
  --embedding-dim=512 \
  --save-dir=models \
  --sample-rate=16000
```

## Model Testing

The `test_classifier.py` script provides comprehensive evaluation of your trained model on test data with support for both encoder types.

### Basic Usage - Zipformer
```bash
python train/test_classifier.py \
  --test-labels data/test_labels.txt \
  --model-path models/best_model.pt \
  --encoder-type zipformer \
  --encoder-model ./encoder_model/encoder-epoch-28-avg-13.onnx \
  --decoder-model ./encoder_model/decoder-epoch-28-avg-13.onnx \
  --joiner-model ./encoder_model/joiner-epoch-28-avg-13.onnx
```

### Basic Usage - FastConformer
```bash
python train/test_classifier.py \
  --test-labels data/test_labels.txt \
  --model-path models/best_model.pt \
  --encoder-type fastconformer \
  --encoder-model nvidia/stt_kk_ru_fastconformer_hybrid_large
```

### Advanced Testing

#### FastConformer with Custom Threshold
```bash
python train/test_classifier.py \
  --test-labels data/test_labels.txt \
  --model-path models/best_model.pt \
  --encoder-type fastconformer \
  --encoder-model nvidia/stt_kk_ru_fastconformer_hybrid_large \
  --threshold 0.7 \
  --output-dir test_results
```

#### Zipformer with Custom Settings
```bash
python train/test_classifier.py \
  --test-labels data/test_labels.txt \
  --model-path models/best_model.pt \
  --encoder-type zipformer \
  --encoder-model ./encoder_model/encoder-epoch-28-avg-13.onnx \
  --decoder-model ./encoder_model/decoder-epoch-28-avg-13.onnx \
  --joiner-model ./encoder_model/joiner-epoch-28-avg-13.onnx \
  --threshold 0.7 \
  --output-dir test_results
```

### Testing with Wav2Vec Models

```bash
python train/test_classifier.py \
  --test-labels data/test_labels.txt \
  --model-path models/best_model.pt \
  --encoder-type wav2vec \
  --encoder-model facebook/wav2vec2-xls-r-300m
```

### Command Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--test-labels` | ✅ | - | File containing test labels with full paths |
| `--model-path` | ✅ | - | Path to the trained model checkpoint |
| `--encoder-type` | ❌ | zipformer | Type of encoder: 'zipformer', 'fastconformer', or 'wav2vec' |
| `--encoder-model` | ❌ | None | Path/name of encoder model |
| `--decoder-model` | ❌ | None | Path to decoder ONNX model (zipformer only) |
| `--joiner-model` | ❌ | None | Path to joiner ONNX model (zipformer only) |
| `--threshold` | ❌ | 0.7 | Probability threshold for classification |
| `--batch-size` | ❌ | 32 | Batch size for testing |
| `--sample-rate` | ❌ | 16000 | Sample rate of audio files |
| `--output-dir` | ❌ | test_results | Directory to save test results |
| `--num-workers` | ❌ | 4 | Number of workers for data loading |

### Test Data Format

The test labels file should contain one line per audio file:
```
/path/to/audio1.wav 0
/path/to/audio2.wav 1
/path/to/audio3.wav 0
```

### Output Files

The testing script generates several output files in the specified output directory:

#### 1. Test Metrics (`test_metrics.txt`)
Summary of all evaluation metrics:
```
Loss: 0.3456
Accuracy: 85.67%
Precision: 0.8234
Recall: 0.7891
F1 Score: 0.8059
AUC: 0.9123

Confusion Matrix:
[[TN, FP]
 [FN, TP]]
[[145  23]
 [ 18  89]]
```

#### 2. Detailed Results (`test_results.csv`)
Complete results for every test file:
```csv
audio_file,true_label,predicted_label,probability,true_label_text,predicted_label_text,is_correct
/path/audio1.wav,0,0,0.2341,not_machine,not_machine,True
/path/audio2.wav,1,1,0.8567,answering_machine,answering_machine,True
/path/audio3.wav,0,1,0.7234,not_machine,answering_machine,False
```

#### 3. Misclassified Files (`misclassified.csv`)
Detailed analysis of incorrectly classified files for error analysis and model improvement.

### Key Features

- **Configurable Threshold**: Adjust classification threshold based on your precision/recall requirements
- **Comprehensive Metrics**: Get all standard binary classification metrics
- **Error Analysis**: Detailed breakdown of misclassified files
- **Confusion Matrix**: Visual representation of classification performance
- **Export Results**: All results saved in CSV format for further analysis
- **ONNX Support**: Compatible with encoder models for feature extraction

### Threshold Selection

The classification threshold affects the precision/recall trade-off:
- **Lower threshold (e.g., 0.3)**: Higher recall, lower precision (catches more answering machines)
- **Higher threshold (e.g., 0.8)**: Higher precision, lower recall (more confident predictions)
- **Default (0.7)**: Balanced approach

### Interpreting Results

#### Confusion Matrix
```
[[TN, FP]    True Negatives: Correctly identified non-machines
 [FN, TP]]   False Positives: Non-machines classified as machines
             False Negatives: Machines classified as non-machines  
             True Positives: Correctly identified machines
```

#### Key Metrics
- **Accuracy**: Overall correctness
- **Precision**: How reliable are positive predictions
- **Recall**: How many actual positives were found
- **F1 Score**: Balance between precision and recall
- **AUC**: Model's ability to distinguish between classes

## Audio Inference

The `audio_inference.py` script provides batch processing capabilities for classifying audio files using a trained model with support for both Zipformer and FastConformer encoders.

### Basic Usage - FastConformer
```bash
python audio_inference.py \
  --input-file data/audio_paths.txt \
  --encoder-type fastconformer \
  --encoder-model nvidia/stt_kk_ru_fastconformer_hybrid_large
```

### Basic Usage - Zipformer
```bash
python audio_inference.py \
  --input-file data/audio_paths.txt \
  --encoder-type zipformer \
  --encoder-model models/encoder.onnx \
  --decoder-model models/decoder.onnx \
  --joiner-model models/joiner.onnx
```

### Advanced Usage

#### FastConformer with Custom Model and Threshold
```bash
python audio_inference.py \
  --input-file data/audio_paths.txt \
  --encoder-type fastconformer \
  --encoder-model nvidia/stt_en_fastconformer_hybrid_large_streaming \
  --model-path models/my_model.pt \
  --threshold 0.7 \
  --output-dir results
```

#### FastConformer with Local .nemo File
```bash
python audio_inference.py \
  --input-file data/audio_paths.txt \
  --encoder-type fastconformer \
  --encoder-model path/to/your/model.nemo \
  --threshold 0.6
```

#### Using CPU Instead of GPU
```bash
python audio_inference.py \
  --input-file data/audio_paths.txt \
  --encoder-type fastconformer \
  --encoder-model nvidia/stt_kk_ru_fastconformer_hybrid_large \
  --device cpu
```

#### Zipformer with Custom Audio Settings
```bash
python audio_inference.py \
  --input-file data/audio_paths.txt \
  --encoder-type zipformer \
  --encoder-model models/encoder.onnx \
  --decoder-model models/decoder.onnx \
  --joiner-model models/joiner.onnx \
  --sample-rate 8000 \
  --min-duration 5.0
```

### Inference with Wav2Vec Models

```bash
python train/infer_classifier.py \
  --input-file data/test_files.txt \
  --model-path models/best_model.pt \
  --encoder-type wav2vec \
  --encoder-model facebook/wav2vec2-xls-r-300m
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input-file` | `data/data_paths.txt` | File containing paths to audio files to process |
| `--model-path` | `./classifier_model/best_model.pt` | Path to the trained classifier model |
| `--encoder-type` | `zipformer` | Type of encoder: 'zipformer', 'fastconformer', or 'wav2vec' |
| `--encoder-model` | `./encoder_model/encoder-epoch-28-avg-13.onnx` | Path to encoder model (ONNX for zipformer, model name/path for fastconformer) |
| `--decoder-model` | `./encoder_model/decoder-epoch-28-avg-13.onnx` | Path to decoder ONNX model (zipformer only) |
| `--joiner-model` | `./encoder_model/joiner-epoch-28-avg-13.onnx` | Path to joiner ONNX model (zipformer only) |
| `--detected-file` | `inference_results/detected_answering_machines.txt` | Output file for detected answering machines |
| `--not-machine-file` | `inference_results/not_machine.txt` | Output file for non-answering machines |
| `--too-short-file` | `inference_results/too_short_audios.txt` | Output file for audio files that are too short |
| `--threshold` | `0.50` | Classification threshold (0.0-1.0) |
| `--sample-rate` | `16000` | Sample rate for audio processing |
| `--min-duration` | `3.0` | Minimum audio duration in seconds |
| `--device` | `cuda` | Device to use for inference (cuda/cpu) |
| `--batch-report-interval` | `50` | Report statistics every N processed files |

### Input File Format

The input file should contain one audio file path per line:
```
/path/to/audio1.wav
/path/to/audio2.mp3
/path/to/audio3.flac
```

### Output Files

The script generates three output files:
- **Detected answering machines**: Audio files classified as answering machines
- **Not machines**: Audio files classified as human voices or other non-answering machine audio
- **Too short**: Audio files shorter than the minimum duration threshold

### Features

- **Automatic resume**: The script automatically skips already processed files
- **Memory management**: Aggressive memory cleanup for processing large datasets
- **Progress tracking**: Real-time statistics and progress reporting
- **Error handling**: Graceful handling of corrupted files and memory issues
- **Batch processing**: Efficient processing of large audio datasets

## Components

### AudioClassifierDataset

Handles loading audio files and their corresponding labels with support for both encoder types.

### AudioClassifierDataLoader

Handles batch processing with encoder-specific logic:
- **Zipformer**: Batch processing with Kaldi Fbank feature extraction
- **FastConformer**: Individual file processing with NeMo preprocessing pipeline
- Automatic feature extraction and embedding generation
- Support for different encoder architectures

### AudioClassifier

A binary neural network classifier with:
- Configurable input dimension (based on encoder output or feature size)
- 2 hidden layers with configurable dimensions
- Dropout and batch normalization for regularization
- Output layer with 2 classes (0 = not answermachine, 1 = answermachine)
- Probability prediction method

## FastConformer Models

### Using Custom .nemo Models

# Inference with custom .nemo model  
python audio_inference.py \
  --encoder-type fastconformer \
  --encoder-model /path/to/your/custom_model.nemo \
  --input-file data/audio_paths.txt
```
