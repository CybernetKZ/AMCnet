"""
Audio Inference Script for Answering Machine Detection

This script processes audio files to detect answering machines using a pre-trained classifier.
Supports both Zipformer (ONNX) and FastConformer (NeMo) encoders.

Usage examples:
    # default
    python audio_inference.py --input-file data/audio_paths.txt

    # Using FastConformer encoder
    python audio_inference.py --input-file data/audio_paths.txt \
        --encoder-type fastconformer \
        --encoder-model nvidia/stt_kk_ru_fastconformer_hybrid_large

    # Using FastConformer with local .nemo file
    python audio_inference.py --input-file data/audio_paths.txt \
        --encoder-type fastconformer \
        --encoder-model path/to/model.nemo

    # Custom threshold and output directory
    python audio_inference.py --input-file data/audio_paths.txt --threshold 0.7 --output-dir results

    # Use CPU instead of GPU
    python audio_inference.py --input-file data/audio_paths.txt --device cpu

    # Custom zipformer model paths
    python audio_inference.py --input-file data/audio_paths.txt \
        --encoder-type zipformer \
        --model-path models/my_model.pt \
        --encoder-model models/encoder.onnx \
        --decoder-model models/decoder.onnx \
        --joiner-model models/joiner.onnx

    # Custom minimum duration and sample rate
    python audio_inference.py --input-file data/audio_paths.txt \
        --min-duration 5.0 --sample-rate 8000
"""

import os
import logging
import torch
import torch.nn as nn
import numpy as np
import tqdm
import kaldifeat
import math
from torch.nn.utils.rnn import pad_sequence
import torchaudio
import gc
import argparse

from train.classifier_model import AudioClassifier
from infernece.base_model import OnnxModel
from nemo.collections import asr as nemo_asr

def setup_logger():
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    return logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Audio inference for answering machine detection")
    parser.add_argument("--input-file", type=str, default="data/data_paths.txt", 
                       help="File containing paths to audio files to process")
    parser.add_argument("--model-path", type=str, default="./classifier_model/best_model.pt",
                       help="Path to the trained classifier model")
    parser.add_argument("--encoder-type", type=str, default="zipformer", 
                       help="Type of encoder - 'zipformer' or 'fastconformer'")
    parser.add_argument("--encoder-model", type=str, default="./encoder_model/encoder-epoch-28-avg-13.onnx",
                       help="Path to encoder ONNX model (zipformer) or model name/path (fastconformer)")
    parser.add_argument("--decoder-model", type=str, default="./encoder_model/decoder-epoch-28-avg-13.onnx",
                       help="Path to decoder ONNX model (zipformer only)")
    parser.add_argument("--joiner-model", type=str, default="./encoder_model/joiner-epoch-28-avg-13.onnx",
                       help="Path to joiner ONNX model (zipformer only)")
    parser.add_argument("--output-dir", type=str, default="inference_results",
                       help="Directory to save inference results")
    parser.add_argument("--detected-file", type=str, default="inference_results/detected_answering_machines.txt",
                       help="File to save detected answering machines")
    parser.add_argument("--not-machine-file", type=str, default="inference_results/not_machine.txt",
                       help="File to save non-answering machines")
    parser.add_argument("--too-short-file", type=str, default="inference_results/too_short_audios.txt",
                       help="File to save too short audio files")
    parser.add_argument("--threshold", type=float, default=0.50,
                       help="Classification threshold")
    parser.add_argument("--sample-rate", type=int, default=16000,
                       help="Sample rate for audio processing")
    parser.add_argument("--min-duration", type=float, default=3.0,
                       help="Minimum audio duration in seconds")
    parser.add_argument("--batch-report-interval", type=int, default=50,
                       help="Report statistics every N processed files")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use for inference (cuda/cpu)")
    return parser.parse_args()

def clear_memory():
    """Clear CUDA cache and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    gc.collect()

def process_single_audio_inference(audio_path, model, encoder, encoder_type, device, threshold, sample_rate=16000, min_duration=3):
    """Process a single audio file and return prediction results."""
    logger = logging.getLogger(__name__)
    
    wave = None
    features = None
    encoder_out = None
    encoder_out_lens = None
    embedding = None
    embedding_cpu = None
    embedding_np = None
    embedding_tensor = None
    outputs = None
    probs = None
    resampler = None
    fbank = None

    try:

        wave, sr = torchaudio.load(audio_path)
        duration = wave.shape[1] / sr
        if duration < min_duration:
            logger.warning(f"Audio duration ({duration:.2f}s) is less than minimum duration ({min_duration}s). Skipping...")
            raise RuntimeError(f"Audio too short: {duration:.2f}s")
        
        if encoder_type == "fastconformer":
            
            from infernece.fastconformer_encoder import run_encoder
            
            with torch.no_grad():
                try:
                    
                    encoder_out = run_encoder(
                        audio_path=audio_path,
                        model=encoder,
                        target_sample_rate=sample_rate,
                        normalize=True
                    )
                    
                    embedding = encoder_out.mean(dim=2).squeeze(0)  
                    
                    del encoder_out
                    encoder_out = None
                    clear_memory()
                    
                    embedding_tensor = embedding.unsqueeze(0).to(device)
                    del embedding
                    embedding = None
                    
                    outputs = model(embedding_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    prob = probs[0][1].item() 
                    pred = 1 if prob > threshold else 0
                    
                    del embedding_tensor, outputs, probs
                    embedding_tensor = None
                    outputs = None
                    probs = None

                    return pred, prob
                    
                except Exception as e:
                    logger.error(f"Error in FastConformer processing: {str(e)}")
                    raise
                    
        else:
            
            if sr != sample_rate:
                logger.info(f"Resampling from {sr}Hz to {sample_rate}Hz")
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
                wave = resampler(wave)
                del resampler
                resampler = None

            wave = wave.to(device)
            
            opts = kaldifeat.FbankOptions()
            opts.device = "cuda"  
            opts.frame_opts.dither = 0
            opts.frame_opts.snip_edges = False
            opts.frame_opts.samp_freq = sample_rate
            opts.mel_opts.num_bins = 80
            opts.mel_opts.high_freq = -400
            fbank = kaldifeat.Fbank(opts)

            chunk_size = 16000 * 30  
            if wave.shape[1] > chunk_size:
                logger.info(f"Processing audio in chunks of {chunk_size/sr:.1f} seconds")
                features_list = []
                for i in range(0, wave.shape[1], chunk_size):
                    chunk = wave[:, i:i+chunk_size]
                    chunk_features = fbank([chunk[0]])[0]
                    features_list.append(chunk_features)
                    clear_memory()
                
                features = torch.cat(features_list, dim=0)
            else:
                features = fbank([wave[0]])[0]

            if features.shape[0] > 10000:  
                logger.warning(f"Features too long ({features.shape[0]} frames). Trimming...")
                features = features[:10000]

            feature_lengths = torch.tensor([features.size(0)], dtype=torch.int64, device=device)

            del wave, fbank
            wave = None
            fbank = None
            clear_memory()

            features = pad_sequence(
                [features],
                batch_first=True,
                padding_value=math.log(1e-10)
            )

            with torch.no_grad():
                try:
                    encoder_out, encoder_out_lens = encoder.run_encoder(features, feature_lengths)
                    
                    del features, feature_lengths
                    features = None
                    clear_memory()
                    
                    valid_length = encoder_out_lens[0]
                    embedding = encoder_out[0, :valid_length].mean(dim=0)
                    
                    if isinstance(embedding, torch.Tensor):
                        embedding_cpu = embedding.detach().cpu().clone()
                        embedding_np = embedding_cpu.numpy().copy()  
                        del embedding_cpu
                        embedding_cpu = None
                    else:
                        embedding_np = embedding.copy()
                    
                    del encoder_out, encoder_out_lens, embedding
                    encoder_out = None
                    encoder_out_lens = None
                    embedding = None
                    clear_memory()
                    
                    embedding_tensor = torch.from_numpy(embedding_np).unsqueeze(0).to(device)
                    del embedding_np  
                    embedding_np = None
                    
                    outputs = model(embedding_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    prob = probs[0][1].item() 
                    pred = 1 if prob > threshold else 0
                    
                    del embedding_tensor, outputs, probs
                    embedding_tensor = None
                    outputs = None
                    probs = None

                    return pred, prob
                    
                except RuntimeError as e:
                    if "out of memory" in str(e) or "Failed to allocate memory" in str(e):
                        logger.error(f"GPU out of memory while processing {audio_path}. Trying with reduced batch size...")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        raise RuntimeError(f"GPU out of memory: {str(e)}")
                    raise
                except Exception as e:
                    if "Integer overflow" in str(e):
                        logger.error(f"Integer overflow while processing {audio_path}. File might be too large or corrupted.")
                        raise RuntimeError(f"Integer overflow: {str(e)}")
                    raise
        
    finally:
        variables_to_clean = [
            'wave', 'features', 'encoder_out', 'encoder_out_lens', 
            'embedding', 'embedding_cpu', 'embedding_np', 'embedding_tensor', 
            'outputs', 'probs', 'resampler', 'fbank'
        ]
        
        for var_name in variables_to_clean:
            if var_name in locals() and locals()[var_name] is not None:
                del locals()[var_name]
        
        clear_memory()

def process_audio(audio_path, model, encoder, encoder_type, device, threshold, sample_rate=16000, detected_file=None, not_machine_file=None, stats=None, min_duration=3):
    """Process a single audio file with the pre-initialized models."""
    logger = setup_logger()

    if not os.path.exists(audio_path):
        raise ValueError(f"Audio file not found: {audio_path}")

    try:
        logger.info("Running inference...")
        pred, prob = process_single_audio_inference(audio_path, model, encoder, encoder_type, device, threshold, sample_rate, min_duration)

        logger.info("\nResults:")
        logger.info(f"Audio file: {os.path.basename(audio_path)}")
        logger.info(f"Prediction: {'answering_machine' if pred == 1 else 'not_machine'}")
        logger.info(f"Probability: {prob:.4f}")
        
        if stats is not None:
            if pred == 1:
                stats['valid'] += 1
            else:
                stats['invalid'] += 1
            total = stats['valid'] + stats['invalid']
            valid_percent = (stats['valid'] / total) * 100 if total > 0 else 0
            invalid_percent = (stats['invalid'] / total) * 100 if total > 0 else 0
            logger.info(f"\nCurrent Statistics:")
            logger.info(f"Valid (answering machine): {stats['valid']} ({valid_percent:.1f}%)")
            logger.info(f"Invalid (not machine): {stats['invalid']} ({invalid_percent:.1f}%)")
            logger.info(f"Total processed: {total}")

        if pred == 1 and detected_file is not None:
            with open(detected_file, 'a') as f:
                f.write(f"{audio_path}\n")
        elif pred == 0 and not_machine_file is not None:
            with open(not_machine_file, 'a') as f:
                f.write(f"{audio_path}\n")
            

        if prob > 0.9:
            confidence = "Very High"
        elif prob > 0.7:
            confidence = "High"
        elif prob > 0.5:
            confidence = "Moderate"
        else:
            confidence = "Low"
        
        logger.info(f"Confidence: {confidence}")

    except Exception as e:
        logger.error(f"Error processing file {audio_path}: {str(e)}")
        raise
    finally:
        clear_memory()

def load_processed_files(file_path):
    """Load already processed files from a text file."""
    if not os.path.exists(file_path):
        return set()
    with open(file_path, 'r') as f:
        return set(line.strip() for line in f)

def main():
    args = parse_args()
    logger = setup_logger()
    
    
    detected_file = args.detected_file
    not_machine_file = args.not_machine_file
    too_short_file = args.too_short_file
    
    
    for file_path in [detected_file, not_machine_file, too_short_file]:
        file_dir = os.path.dirname(file_path)
        if file_dir and not os.path.exists(file_dir):
            os.makedirs(file_dir)
            logger.info(f"Created directory: {file_dir}")
    
    logger.info(f"Input file: {args.input_file}")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Threshold: {args.threshold}")
    logger.info(f"Sample rate: {args.sample_rate}")
    logger.info(f"Minimum duration: {args.min_duration}s")
    logger.info(f"Detected answering machines will be saved to: {detected_file}")
    logger.info(f"Non-answering machines will be saved to: {not_machine_file}")
    logger.info(f"Too short audios will be saved to: {too_short_file}")

    processed_files = load_processed_files(detected_file) | load_processed_files(not_machine_file) | load_processed_files(too_short_file)
    logger.info(f"Found {len(processed_files)} already processed files")
    
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file {args.input_file} not found")
        
    with open(args.input_file, 'r') as f:
        all_audio_paths = [line.strip() for line in f if line.strip()]
    
    audio_paths = [path for path in all_audio_paths if path not in processed_files]
    logger.info(f"Found {len(all_audio_paths)} total audio files")
    logger.info(f"Excluding {len(processed_files)} already processed files")
    logger.info(f"Remaining {len(audio_paths)} files to process")
    
    if len(audio_paths) == 0:
        logger.info("No new files to process. Exiting.")
        return
    
    stats = {'valid': 0, 'invalid': 0, 'skipped': 0, 'errors': 0, 'too_short': 0}
    
    
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    encoder = None
    model = None
    
    try:
        logger.info(f"Loading model from {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        
        
        if args.encoder_type == "zipformer":
            if all([args.encoder_model, args.decoder_model, args.joiner_model]):
                logger.info("Initializing zipformer encoder model...")
                encoder = OnnxModel(
                    encoder_model_filename=args.encoder_model,
                    decoder_model_filename=args.decoder_model,
                    joiner_model_filename=args.joiner_model,
                )
            else:
                raise ValueError("For zipformer, all encoder, decoder, and joiner model paths are required")
        elif args.encoder_type == "fastconformer":
            logger.info("Initializing fastconformer encoder model...")
            if ".nemo" in args.encoder_model:
                encoder = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from( 
                    restore_path=args.encoder_model,
                    map_location=f"cuda"
                )
            else:
                encoder = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained( 
                    model_name=args.encoder_model,
                    map_location=f"cuda"
                )
        else:
            raise ValueError(f"Unsupported encoder type: {args.encoder_type}. Use 'zipformer' or 'fastconformer'")
        
        model = AudioClassifier(
            input_dim=checkpoint['input_dim'],
            hidden_dims=checkpoint['hidden_dims'],
            num_classes=2
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        del checkpoint
        clear_memory()
        
        for i, audio_path in enumerate(tqdm.tqdm(audio_paths)):
            try:
                process_audio(
                    audio_path=audio_path,
                    model=model,
                    encoder=encoder,
                    encoder_type=args.encoder_type,
                    device=device,
                    threshold=args.threshold,
                    sample_rate=args.sample_rate,
                    detected_file=detected_file,
                    not_machine_file=not_machine_file,
                    stats=stats,
                    min_duration=args.min_duration
                )
                
                if (i + 1) % args.batch_report_interval == 0:
                    logger.info(f"Processed {i + 1} files. Running aggressive memory cleanup...")
                    logger.info(f"Statistics: Valid={stats['valid']}, Invalid={stats['invalid']}, Skipped={stats['skipped']}, Errors={stats['errors']}, Too Short={stats['too_short']}")
                    import ctypes
                    try:
                        libc = ctypes.CDLL("libc.so.6")
                        libc.malloc_trim(0)
                    except:
                        pass
                    clear_memory()
                    
            except RuntimeError as e:
                if "out of memory" in str(e) or "Failed to allocate memory" in str(e):
                    logger.error(f"GPU out of memory while processing {audio_path}. Skipping file...")
                    stats['errors'] += 1
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                elif "Integer overflow" in str(e):
                    logger.error(f"Integer overflow while processing {audio_path}. Skipping file...")
                    stats['errors'] += 1
                    continue
                elif "Audio too short" in str(e):
                    logger.warning(f"Audio file too short: {audio_path}")
                    stats['too_short'] += 1
                    with open(too_short_file, 'a') as f:
                        f.write(f"{audio_path}\n")
                    continue
                raise
            except Exception as e:
                logger.error(f"Error processing file {audio_path}: {str(e)}")
                stats['errors'] += 1
                continue
            finally:
                clear_memory()
        
        logger.info("\nFinal Statistics:")
        logger.info(f"Valid (not machine): {stats['valid']}")
        logger.info(f"Invalid (answering machine): {stats['invalid']}")
        logger.info(f"Skipped (already processed): {len(processed_files)}")
        logger.info(f"Too short (< {args.min_duration}s): {stats['too_short']}")
        logger.info(f"Errors: {stats['errors']}")
        logger.info(f"Total files processed: {len(audio_paths)}")
        logger.info(f"Total files in dataset: {len(all_audio_paths)}")
    
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        raise
    finally:
        if encoder is not None:
            del encoder
        if model is not None:
            del model

        clear_memory()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
