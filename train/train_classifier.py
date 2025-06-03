import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import datetime

from dataloader import AudioClassifierDataset
from encoder_dataloader import AudioClassifierDataLoader
from classifier_model import AudioClassifier
from infernece.base_model import OnnxModel

def setup_logger():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    return logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train audio classifier for answermachine detection")
    parser.add_argument("--train-labels", type=str, required=True, help="File containing training labels with full paths")
    parser.add_argument("--val-labels", type=str, required=True, help="File containing validation labels with full paths")
    parser.add_argument("--encoder-model", type=str, default="./cluster_model/encoder-epoch-28-avg-13.onnx", help="Path to encoder ONNX model")
    parser.add_argument("--decoder-model", type=str, default="./cluster_model/decoder-epoch-28-avg-13.onnx", help="Path to decoder ONNX model")
    parser.add_argument("--joiner-model", type=str, default="./cluster_model/joiner-epoch-28-avg-13.onnx", help="Path to joiner ONNX model")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--hidden-dims", type=str, default="1024,512,256", help="Comma-separated list of hidden layer dimensions")
    parser.add_argument("--embedding-dim", type=int, default=512, help="Dimension of audio embeddings")
    parser.add_argument("--save-dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Sample rate of audio files")
    parser.add_argument("--class-weights", type=str, default=None, help="Comma-separated class weights for handling imbalanced data")
    parser.add_argument("--tensorboard-dir", type=str, default="runs", help="Directory for TensorBoard logs")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for data loading")
    return parser.parse_args()

def load_data(labels_file):
    """
    Load data from a label file containing full paths to audio files.
    
    Args:
        labels_file (str): Path to the label file
        
    Returns:
        tuple: (audio_files, labels) where audio_files contains full paths
    """
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

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        if len(batch) == 2:  
            inputs, targets = batch
            inputs = inputs.to(device)
        else:  
            inputs, lengths, targets = batch
            inputs = inputs.to(device)
        
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        all_probs.extend(model.predict_proba(inputs).detach().cpu().numpy())
        
        pbar.set_postfix({
            "loss": total_loss / (pbar.n + 1),
            "acc": 100. * correct / total
        })
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='binary', pos_label=1
    )
    
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except:
        auc = 0.0
    
    metrics = {
        'loss': total_loss / len(loader),
        'acc': 100. * correct / total,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }
    
    return metrics

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for batch in loader:
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
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(model.predict_proba(inputs).detach().cpu().numpy())
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='binary', pos_label=1
    )
    
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except:
        auc = 0.0
    
    metrics = {
        'loss': total_loss / len(loader),
        'acc': 100. * correct / total,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }
    
    return metrics

def main():
    args = parse_args()
    logger = setup_logger()
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    tensorboard_dir = os.path.join(args.tensorboard_dir, f"run_{timestamp}")
    writer = SummaryWriter(tensorboard_dir)
    logger.info(f"TensorBoard logs will be saved to {tensorboard_dir}")
    
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your PyTorch installation.")
    
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True  
    torch.backends.cudnn.deterministic = False  
    
    logger.info(f"Using device: {device}")
    logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA Version: {torch.version.cuda}")
    
    logger.info("Loading training data...")
    train_audio_files, train_labels = load_data(args.train_labels)
    logger.info(f"Loaded {len(train_audio_files)} training files with classes: 0={train_labels.count(0)}, 1={train_labels.count(1)}")
    
    logger.info("Loading validation data...")
    val_audio_files, val_labels = load_data(args.val_labels)
    logger.info(f"Loaded {len(val_audio_files)} validation files with classes: 0={val_labels.count(0)}, 1={val_labels.count(1)}")
    
    encoder_model = None
    if args.encoder_model and args.decoder_model and args.joiner_model:
        logger.info("Initializing encoder model...")
        encoder_model = OnnxModel(
            encoder_model_filename=args.encoder_model,
            decoder_model_filename=args.decoder_model,
            joiner_model_filename=args.joiner_model,
        )
    
    logger.info("Creating datasets...")
    train_dataset = AudioClassifierDataset(train_audio_files, train_labels, sample_rate=args.sample_rate)
    val_dataset = AudioClassifierDataset(val_audio_files, val_labels, sample_rate=args.sample_rate)
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    logger.info("Creating dataloaders...")
    train_loader = AudioClassifierDataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        encoder_model=encoder_model
    )
    
    val_loader = AudioClassifierDataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        encoder_model=encoder_model
    )
    
    logger.info("Creating binary classifier model for answermachine detection...")
    hidden_dims = [int(dim) for dim in args.hidden_dims.split(",")]
    
    if encoder_model:
        input_dim = args.embedding_dim
    else:
        input_dim = 512
    
    model = AudioClassifier(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=2
    )
    model = model.to(device)
    
    if args.class_weights:
        weights = [float(w) for w in args.class_weights.split(",")]
        class_weights = torch.tensor(weights, device=device)
        logger.info(f"Using class weights: {weights}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

   
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    logger.info("Starting training...")
    best_f1 = 0
    
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        
        
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        logger.info(f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.2f}%, " +
                   f"F1: {train_metrics['f1']:.4f}, AUC: {train_metrics['auc']:.4f}")
        
        
        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('Accuracy/train', train_metrics['acc'], epoch)
        writer.add_scalar('F1/train', train_metrics['f1'], epoch)
        writer.add_scalar('AUC/train', train_metrics['auc'], epoch)
        
        
        val_metrics = validate(model, val_loader, criterion, device)
        logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['acc']:.2f}%, " +
                   f"F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")
        
        
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('Accuracy/val', val_metrics['acc'], epoch)
        writer.add_scalar('F1/val', val_metrics['f1'], epoch)
        writer.add_scalar('AUC/val', val_metrics['auc'], epoch)
        
        
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'input_dim': input_dim,
            'hidden_dims': hidden_dims
        }
        checkpoint_path = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            logger.info(f"New best F1 score: {val_metrics['f1']:.4f}")
            best_model_path = os.path.join(args.save_dir, "best_model.pt")
            torch.save(checkpoint, best_model_path)
            logger.info(f"Saved best model to {best_model_path}")
    
    
    final_model_path = os.path.join(args.save_dir, "final_model.pt")
    torch.save(checkpoint, final_model_path)
    logger.info(f"Saved final model to {final_model_path}")
    
    
    writer.close()
    logger.info(f"Training completed. Best validation F1 score: {best_f1:.4f}")
    logger.info(f"To view TensorBoard logs, run: tensorboard --logdir={tensorboard_dir}")


if __name__ == "__main__":
    main()