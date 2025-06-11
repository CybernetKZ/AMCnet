import torch
import librosa
import numpy as np
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from typing import Optional, Union
import warnings
import os

class Wav2VecEncoder:
    def __init__(self, model_name: str = "facebook/wav2vec2-xls-r-300m", device: str = "cuda"):
        """
        Initialize Wav2Vec encoder.
        
        Args:
            model_name: Pretrained model name from HuggingFace
            device: Device to run the model on
        """
        self.device = device
        self.model_name = model_name
        
        
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.model = self.model.to(device)
        self.model.eval()
        
        
        self.feature_dim = self.model.config.hidden_size
        
    def extract_features(self, audio_path: str, target_sample_rate: int = 16000) -> torch.Tensor:
        """
        Extract features from audio file using Wav2Vec.
        
        Args:
            audio_path: Path to audio file
            target_sample_rate: Target sample rate
            
        Returns:
            Encoded features tensor
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            
            audio, sr = librosa.load(audio_path, sr=target_sample_rate)
            
            
            inputs = self.feature_extractor(
                audio, 
                sampling_rate=target_sample_rate, 
                return_tensors="pt",
                padding=True
            )
            
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                features = outputs.last_hidden_state  
            
            return features
            
        except Exception as e:
            raise RuntimeError(f"Error processing audio file {audio_path}: {str(e)}")

def run_encoder(audio_path: str, 
                model: Wav2VecEncoder,
                target_sample_rate: Optional[int] = 16000,
                normalize: bool = True,
                device: Optional[str] = None) -> torch.Tensor:
    """
    Run wav2vec encoder on an audio file.
    Compatible with your existing encoder interface.
    
    Args:
        audio_path: Path to the audio file
        model: Wav2VecEncoder instance
        target_sample_rate: Target sample rate for resampling
        normalize: Whether to normalize (kept for compatibility)
        device: Device to run on (if None, uses model.device)
    
    Returns:
        Encoded audio tensor
    """
    return model.extract_features(audio_path, target_sample_rate) 