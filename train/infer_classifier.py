import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from train.dataloader import AudioClassifierDataset
from train.encoder_dataloader import AudioClassifierDataLoader
from train.classifier_model import AudioClassifier
from infernece.encoder_classifier_model import OnnxModel

def setup_logger():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    return logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on audio files for answering machine detection")
    parser.add_argument("--input-file", type=str, required=True, help="File containing paths to audio files")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--encoder-type", type=str, default="zipformer", help="Type of encoder - 'zipformer' or 'fastconformer'")
    parser.add_argument("--encoder-model", type=str, default=None, help="Path to encoder ONNX model")
    parser.add_argument("--decoder-model", type=str, default=None, help="Path to decoder ONNX model")
    parser.add_argument("--joiner-model", type=str, default=None, help="Path to joiner ONNX model")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Sample rate of audio files")
    parser.add_argument("--output-dir", type=str, default="inference_results", help="Directory to save inference results")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--threshold", type=float, default=0.7, help="Probability threshold for answering machine classification")
    return parser.parse_args()

def load_audio_files(input_file):
    """Load audio file paths from input file."""
    audio_files = []
    
    with open(input_file, "r") as f:
        for line in f:
            file_path = line.strip()
            if not os.path.exists(file_path):
                raise ValueError(f"Audio file not found: {file_path}")
            audio_files.append(file_path)
    
    return audio_files

def infer(model, loader, device, threshold=0.7):
    """Run inference on audio files and return predictions."""
    model.eval()
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Processing"):
            if len(batch) == 2:
                inputs, _ = batch
                inputs = inputs.to(device)
            else:
                inputs, lengths, _ = batch
                inputs = inputs.to(device)
            
            outputs = model(inputs)
            
            
            probs = torch.softmax(outputs, dim=1)[:, 1]
            
            predicted = (probs > threshold).long()
            
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return all_preds, all_probs

def main():
    args = parse_args()
    logger = setup_logger()
    
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    
    logger.info(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    
    encoder_model = None
    if all([args.encoder_model, args.decoder_model, args.joiner_model]):
        logger.info("Initializing encoder model...")
        encoder_model = OnnxModel(
            encoder_model_filename=args.encoder_model,
            decoder_model_filename=args.decoder_model,
            joiner_model_filename=args.joiner_model,
        )
    
    
    model = AudioClassifier(
        input_dim=checkpoint['input_dim'],
        hidden_dims=checkpoint['hidden_dims'],
        num_classes=2
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    
    logger.info("Loading audio files...")
    audio_files = load_audio_files(args.input_file)
    logger.info(f"Loaded {len(audio_files)} audio files")
    
    
    test_dataset = AudioClassifierDataset(audio_files, [0] * len(audio_files), sample_rate=args.sample_rate)  
    test_loader = AudioClassifierDataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        encoder_model=encoder_model,
        encoder_type=args.encoder_type,
        num_workers=args.num_workers
    )
    
    
    logger.info(f"Running inference with threshold {args.threshold}...")
    predictions, probabilities = infer(model, test_loader, device, args.threshold)
    
    
    predictions_file = os.path.join(args.output_dir, 'predictions.txt')
    logger.info(f"Saving predictions to {predictions_file}")
    with open(predictions_file, 'w') as f:
        for file_path, pred, prob in zip(audio_files, predictions, probabilities):
            f.write(f"{file_path} {pred} {prob:.4f}\n")
    
    
    answering_machines_file = os.path.join(args.output_dir, 'answering_machines.txt')
    logger.info(f"Saving answering machine paths to {answering_machines_file}")
    with open(answering_machines_file, 'w') as f:
        for file_path, pred in zip(audio_files, predictions):
            if pred == 1:  
                f.write(f"{file_path}\n")
    
    
    num_machines = sum(predictions)
    logger.info("\nInference Summary:")
    logger.info(f"Total files processed: {len(audio_files)}")
    logger.info(f"Answering machines detected: {num_machines} ({num_machines/len(audio_files)*100:.1f}%)")
    logger.info(f"Non-answering machines: {len(audio_files) - num_machines} ({(len(audio_files) - num_machines)/len(audio_files)*100:.1f}%)")
    
    
    summary_file = os.path.join(args.output_dir, 'summary.txt')
    logger.info(f"Saving summary to {summary_file}")
    with open(summary_file, 'w') as f:
        f.write(f"Total files processed: {len(audio_files)}\n")
        f.write(f"Answering machines detected: {num_machines} ({num_machines/len(audio_files)*100:.1f}%)\n")
        f.write(f"Non-answering machines: {len(audio_files) - num_machines} ({(len(audio_files) - num_machines)/len(audio_files)*100:.1f}%)\n")

if __name__ == "__main__":
    main() 