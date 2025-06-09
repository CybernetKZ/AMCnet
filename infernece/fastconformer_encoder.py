import torch
import numpy as np
from pydub import AudioSegment
import os
from typing import Optional, Union
import warnings
from nemo.collections import asr as nemo_asr


model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained( 
    model_name="nvidia/stt_kk_ru_fastconformer_hybrid_large",
    map_location=f"cuda"
)

def run_encoder(audio_path: str, 
                model: torch.nn.Module,
                target_sample_rate: Optional[int] = None,
                normalize: bool = True,
                device: Optional[str] = None) -> torch.Tensor:
    """
    Run audio encoder on an audio file.
    
    Args:
        audio_path: Path to the audio file
        model: The model with encoder attribute
        target_sample_rate: Target sample rate for resampling (if None, uses original)
        normalize: Whether to normalize audio to [-1, 1] range
        device: Device to run on (if None, uses model.device)
    
    Returns:
        Encoded audio tensor
    """
    
    
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    if not hasattr(model, 'encoder'):
        raise AttributeError("Model must have an 'encoder' attribute")
    
    try:
        
        audio = AudioSegment.from_file(audio_path)  
        
        
        if audio.channels > 1:
            audio = audio.set_channels(1)
            warnings.warn("Audio converted from stereo to mono")
        
        
        if target_sample_rate and audio.frame_rate != target_sample_rate:
            audio = audio.set_frame_rate(target_sample_rate)
            print(f"Audio resampled from {audio.frame_rate} to {target_sample_rate} Hz")
        
        
        audio_samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        
        
        if normalize:
            if audio.sample_width == 2:  
                audio_samples = audio_samples / 32768.0
            elif audio.sample_width == 4:  
                audio_samples = audio_samples / 2147483648.0
            else:
                
                max_val = np.max(np.abs(audio_samples))
                if max_val > 0:
                    audio_samples = audio_samples / max_val
        
        
        audio_tensor = torch.from_numpy(audio_samples).float()
        
        
        
        audio_tensor = audio_tensor.unsqueeze(0)  
        
        
        target_device = device or (model.device if hasattr(model, 'device') else 'cpu')
        audio_tensor = audio_tensor.to(target_device)
        
        
        with torch.no_grad():
            
            audio_length = torch.tensor([audio_tensor.shape[1]], dtype=torch.long, device=audio_tensor.device)
            
            
            processed_signal, processed_length = model.preprocessor(
                input_signal=audio_tensor, 
                length=audio_length
            )
            
            
            encoded, _ = model.encoder(audio_signal=processed_signal, length=processed_length)
        
        return encoded
        
    except Exception as e:
        raise RuntimeError(f"Error processing audio file {audio_path}: {str(e)}")


if __name__ == "__main__":
    encoded_audio = run_encoder(
        '/mnt/nfs/mahmoud/ASR/Egypt/pool_5_ARB/32189af34e1d42f7.wav',
        model=model,
        target_sample_rate=16000,  
        normalize=True
    )
    print(encoded_audio.shape)
