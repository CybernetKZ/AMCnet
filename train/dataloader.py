import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
import kaldifeat

class AudioClassifierDataset(Dataset):
    def __init__(self, audio_files, labels, sample_rate=16000):
        """
        Args:
            audio_files (list): List of audio file paths
            labels (list): List of corresponding labels
            sample_rate (int): Expected sample rate for audio files
        """
        self.audio_files = audio_files
        self.labels = labels
        self.sample_rate = sample_rate
        
        
        opts = kaldifeat.FbankOptions()
        opts.device = "cpu"
        opts.frame_opts.dither = 0
        opts.frame_opts.snip_edges = False
        opts.frame_opts.samp_freq = sample_rate
        opts.mel_opts.num_bins = 80
        opts.mel_opts.high_freq = -400
        self.fbank = kaldifeat.Fbank(opts)
        
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, index):
        
        audio_path = self.audio_files[index]
        wave, sample_rate = torchaudio.load(audio_path)
        
        
        if sample_rate != self.sample_rate:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
            wave = resampler(wave)
        
        
        wave = wave[0]
        features = self.fbank([wave])[0]  
        
        return features, self.labels[index]
