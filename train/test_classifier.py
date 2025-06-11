import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
import numpy as np

from train.dataloader import AudioClassifierDataset
from train.encoder_dataloader import AudioClassifierDataLoader
from train.classifier_model import AudioClassifier
from infernece.encoder_classifier_model import OnnxModel
from nemo.collections import asr as nemo_asr

def setup_logger():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    return logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Test audio classifier for answermachine detection")
    parser.add_argument("--test-labels", type=str, required=True, help="File containing test labels with full paths")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--encoder-type", type=str, default="zipformer", help="Type of encoder - 'zipformer', 'fastconformer', or 'wav2vec'")
    parser.add_argument("--encoder-model", type=str, default=None, help="Path to encoder ONNX model (zipformer), model name/path (fastconformer), or HuggingFace model name (wav2vec)")
    parser.add_argument("--decoder-model", type=str, default=None, help="Path to decoder ONNX model")
    parser.add_argument("--joiner-model", type=str, default=None, help="Path to joiner ONNX model")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Sample rate of audio files")
    parser.add_argument("--output-dir", type=str, default="test_results", help="Directory to save test results")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--threshold", type=float, default=0.7, help="Probability threshold for answering machine classification (default: 0.7)")
    return parser.parse_args()

def load_data(labels_file):
    """Load data from a label file containing full paths to audio files."""
    audio_files = []
    labels = []
    
    with open(labels_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                file_path = parts[0]
                label = int(parts[1])
                if label not in [0, 1]:
                    raise ValueError(f"Label must be 0 (not answermachine) or 1 (answermachine), got {label} for {file_path}")
                if not os.path.exists(file_path):
                    raise ValueError(f"Audio file not found: {file_path}")
                audio_files.append(file_path)
                labels.append(label)
    
    return audio_files, labels

def test(model, loader, criterion, device, threshold=0.7):
    """Run inference on test data and compute metrics."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    all_preds = []
    all_targets = []
    all_probs = []
    all_files = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing"):
            if len(batch) == 2:
                inputs, targets = batch
                inputs = inputs.to(device)
            else:
                inputs, lengths, targets = batch
                inputs = inputs.to(device)
            
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            
            
            probs = torch.softmax(outputs, dim=1)[:, 1]
            
            predicted = (probs > threshold).long()
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='binary', pos_label=1
    )
    
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except:
        auc = 0.0
    
    
    cm = confusion_matrix(all_targets, all_preds)
    
    metrics = {
        'loss': total_loss / len(loader),
        'acc': 100. * correct / total,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm
    }
    
    return metrics, all_preds, all_targets, all_probs

def main():
    args = parse_args()
    logger = setup_logger()
    
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    
    logger.info(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    
    encoder_model = None
    if args.encoder_type == "wav2vec":
        if args.encoder_model:
            logger.info("Initializing wav2vec encoder model...")
            from infernece.wav2vec_encoder import Wav2VecEncoder
            encoder_model = Wav2VecEncoder(
                model_name=args.encoder_model,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            raise ValueError("For wav2vec, encoder model path/name is required")
    elif args.encoder_type == "zipformer":
        if args.encoder_model and args.decoder_model and args.joiner_model:
            logger.info("Initializing zipformer encoder model...")
            encoder_model = OnnxModel(
                encoder_model_filename=args.encoder_model,
                decoder_model_filename=args.decoder_model,
                joiner_model_filename=args.joiner_model,
            )
        else:
            raise ValueError("For zipformer, all encoder, decoder, and joiner model paths are required")
    elif args.encoder_type == "fastconformer":
        if args.encoder_model:
            logger.info("Initializing fastconformer encoder model...")
            if ".nemo" not in args.encoder_model:
                encoder_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained( 
                    model_name=args.encoder_model,
                    map_location=f"cuda"
                )
            else:
                encoder_model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from( 
                    restore_path=args.encoder_model,
                    map_location=f"cuda"
                )
        else:
            raise ValueError("For fastconformer, encoder model path is required")
    else:
        raise ValueError(f"Unsupported encoder type: {args.encoder_type}. Use 'zipformer', 'fastconformer', or 'wav2vec'")
    
    
    model = AudioClassifier(
        input_dim=checkpoint['input_dim'],
        hidden_dims=checkpoint['hidden_dims'],
        num_classes=2
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    
    logger.info("Loading test data...")
    test_audio_files, test_labels = load_data(args.test_labels)
    logger.info(f"Loaded {len(test_audio_files)} test files with classes: 0={test_labels.count(0)}, 1={test_labels.count(1)}")
    
    
    num_workers = args.num_workers
    if args.encoder_type in ["fastconformer", "wav2vec"]:
        num_workers = 0  # Avoid multiprocessing issues with these encoders
        if args.num_workers > 0:
            logger.warning(f"Setting num_workers=0 for {args.encoder_type} to avoid multiprocessing issues")
    
    test_dataset = AudioClassifierDataset(test_audio_files, test_labels, sample_rate=args.sample_rate)
    test_loader = AudioClassifierDataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        encoder_model=encoder_model,
        encoder_type=args.encoder_type,
        num_workers=num_workers
    )
    
    logger.info(f"Running test with threshold {args.threshold}...")
    criterion = nn.CrossEntropyLoss()
    metrics, predictions, targets, probabilities = test(model, test_loader, criterion, device, args.threshold)
    
    logger.info("\nTest Results:")
    logger.info(f"Loss: {metrics['loss']:.4f}")
    logger.info(f"Accuracy: {metrics['acc']:.2f}%")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1 Score: {metrics['f1']:.4f}")
    logger.info(f"AUC: {metrics['auc']:.4f}")
    
    logger.info("\nConfusion Matrix:")
    logger.info("[[TN, FP]")
    logger.info(" [FN, TP]]")
    logger.info(metrics['confusion_matrix'])
    
    results_df = pd.DataFrame({
        'audio_file': test_audio_files,
        'true_label': targets,
        'predicted_label': predictions,
        'probability': probabilities  
    })
    
    results_df['true_label_text'] = results_df['true_label'].map({0: 'not_machine', 1: 'answering_machine'})
    results_df['predicted_label_text'] = results_df['predicted_label'].map({0: 'not_machine', 1: 'answering_machine'})
    
    results_df['is_correct'] = results_df['true_label'] == results_df['predicted_label']
    misclassified = results_df[~results_df['is_correct']]
    
    misclassified_path = os.path.join(args.output_dir, 'misclassified.csv')
    misclassified.to_csv(misclassified_path, index=False)
    logger.info(f"\nMisclassified files saved to {misclassified_path}")
    
    logger.info("\nMisclassified Files:")
    for _, row in misclassified.iterrows():
        logger.info(f"File: {row['audio_file']}")
        logger.info(f"True Label: {row['true_label_text']}")
        logger.info(f"Predicted Label: {row['predicted_label_text']}")
        logger.info(f"Probability: {row['probability']:.4f}")
        logger.info("---")
    
    results_path = os.path.join(args.output_dir, 'test_results.csv')
    results_df.to_csv(results_path, index=False)
    logger.info(f"\nDetailed results saved to {results_path}")
    
    metrics_path = os.path.join(args.output_dir, 'test_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"Loss: {metrics['loss']:.4f}\n")
        f.write(f"Accuracy: {metrics['acc']:.2f}%\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1']:.4f}\n")
        f.write(f"AUC: {metrics['auc']:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write("[[TN, FP]\n")
        f.write(" [FN, TP]]\n")
        f.write(str(metrics['confusion_matrix']))
    
    logger.info(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    main() 