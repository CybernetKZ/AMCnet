import torch
from torch.utils.data import DataLoader
import torchaudio.transforms as T
from torch.nn.utils.rnn import pad_sequence
import math
import os


class AudioClassifierDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=True, encoder_model=None, encoder_type="zipformer", num_workers=0):
        """
        Args:
            dataset (AudioClassifierDataset): Dataset containing audio files and labels
            batch_size (int): Number of samples per batch
            shuffle (bool): Whether to shuffle the dataset
            encoder_model: Encoder model to extract embeddings (zipformer ONNX, fastconformer, or wav2vec)
            encoder_type (str): Type of encoder - "zipformer", "fastconformer", or "wav2vec"
            num_workers (int): Number of workers for data loading
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.encoder_model = encoder_model
        self.encoder_type = encoder_type.lower()
        
        
        if self.encoder_type not in ["zipformer", "fastconformer", "wav2vec"]:
            raise ValueError("encoder_type must be 'zipformer', 'fastconformer', or 'wav2vec'")
        
        
        if self.encoder_type in ["fastconformer", "wav2vec"]:
            if hasattr(self.dataset, 'return_index'):
                self.dataset.return_index = True
            else:
                raise ValueError(f"Dataset must support return_index=True for {self.encoder_type} encoder. "
                               "Please use the updated AudioClassifierDataset.")
        
        def collate_fn(batch):
            
            if self.encoder_type in ["fastconformer", "wav2vec"]:
                
                features = []
                labels = []
                audio_paths = []
                
                for item in batch:
                    if len(item) == 3:  
                        feature, label, idx = item
                        features.append(feature)
                        labels.append(label)
                        
                        audio_paths.append(self.dataset.audio_files[idx])
                    else:  
                        
                        raise ValueError(f"For {self.encoder_type} encoder, please modify your dataset to return indices. "
                                       "See the updated AudioClassifierDataset.")
            else:
                
                features = [item[0] for item in batch]
                labels = [item[1] for item in batch]
            
            labels_tensor = torch.tensor(labels, dtype=torch.long)
            
            if self.encoder_model is not None:
                embeddings = []
                
                with torch.no_grad():
                    if self.encoder_type == "zipformer":
                        
                        for feature in features:
                            feature_batch = feature.unsqueeze(0)  
                            feature_length = torch.tensor([feature.size(0)], dtype=torch.int64)
                            
                            encoder_out, _ = self.encoder_model.run_encoder(
                                feature_batch, feature_length
                            )
                            
                            emb = encoder_out[0].mean(dim=0)  
                            embeddings.append(emb)
                    
                    elif self.encoder_type == "fastconformer":
                        
                        from infernece.fastconformer_encoder import run_encoder
                        
                        for audio_path in audio_paths:
                            
                            encoder_out = run_encoder(
                                audio_path=audio_path,
                                model=self.encoder_model,
                                target_sample_rate=16000,
                                normalize=True
                            )
                            
                            emb = encoder_out.mean(dim=2)  
                            emb = emb.squeeze(0)  
                            embeddings.append(emb)
                    
                    elif self.encoder_type == "wav2vec":
                        
                        from infernece.wav2vec_encoder import run_encoder
                        
                        for audio_path in audio_paths:
                            
                            encoder_out = run_encoder(
                                audio_path=audio_path,
                                model=self.encoder_model,
                                target_sample_rate=16000,
                                normalize=True
                            )
                            
                            # Average pooling over time dimension
                            emb = encoder_out.mean(dim=1)  # Shape: (batch_size, hidden_size)
                            emb = emb.squeeze(0)  # Remove batch dimension
                            embeddings.append(emb)
                
                
                embeddings = torch.stack(embeddings)  
                return embeddings, labels_tensor
            
            raise ValueError("Encoder model is required for this dataloader")
        
        
        super().__init__(
            dataset=dataset,
            batch_size=batch_size, 
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=num_workers
        )
    
    def __len__(self):
        return len(self.dataset) // self.batch_size + (len(self.dataset) % self.batch_size > 0)
