import torch
import torch.nn as nn

class AudioClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes=2):
        """
        A binary classifier for detecting answer machines in audio
        
        Args:
            input_dim (int): Dimension of input embeddings
            hidden_dims (list): List of dimensions for hidden layers
            num_classes (int): Number of output classes (default: 2 for binary classification)
        """
        super().__init__()


        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.Dropout(0.2))
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the classifier
        
        Args:
            x (torch.Tensor): Input embeddings [batch_size, input_dim]
        
        Returns:
            torch.Tensor: Class logits [batch_size, num_classes]
                          where class 0 = not answermachine, class 1 = answermachine
        """
        return self.classifier(x)
        
    def predict_proba(self, x):
        """
        Get probability of being an answermachine
        
        Args:
            x (torch.Tensor): Input embeddings [batch_size, input_dim]
            
        Returns:
            torch.Tensor: Probability of answermachine class [batch_size, 1]
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)[:, 1] 