import torch
from torch.utils.data import DataLoader
import torchaudio.transforms as T
from torch.nn.utils.rnn import pad_sequence
import math


class AudioClassifierDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=True, encoder_model=None, num_workers=0):
        """
        Args:
            dataset (AudioClassifierDataset): Dataset containing audio files and labels
            batch_size (int): Number of samples per batch
            shuffle (bool): Whether to shuffle the dataset
            encoder_model: Optional encoder model to extract embeddings
            num_workers (int): Number of workers for data loading
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.encoder_model = encoder_model
        
        
        def collate_fn(batch):
            
            features = [item[0] for item in batch]
            labels = [item[1] for item in batch]
            
            
            labels_tensor = torch.tensor(labels, dtype=torch.long)
            
            
            if self.encoder_model is not None:
                embeddings = []
                
                with torch.no_grad():
                    
                    for feature in features:
                        
                        feature_batch = feature.unsqueeze(0)  
                        feature_length = torch.tensor([feature.size(0)], dtype=torch.int64)
                        
                        encoder_out, _ = self.encoder_model.run_encoder(
                            feature_batch, feature_length
                        )
                        
                        emb = encoder_out[0].mean(dim=0)  
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
