import torch
import torch.nn as nn

class NNEU(nn.Module):
    """
    A efficient update neural network.
    """
    
    # the width of the first layer
    in_features = 40960

    def __init__(self):
        super().__init__()

        self.feature_extractor = nn.EmbeddingBag(40961, 256, mode="sum", padding_idx=40960)
        self.fn1 = nn.Linear(2 * 256, 32)
        self.fn2 = nn.Linear(32, 32)
        self.fn3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, white_features, black_features, perspective): 
        """
        Pass the features through the model.
        """
        p = perspective.unsqueeze(1)
        w = self.feature_extractor(white_features)
        b = self.feature_extractor(black_features)
        # the weights switch order corresponding to the different perspectives
        x = torch.cat([torch.where(p, w, b),torch.where(p, w, b)], dim=1)
        x = self.relu(x)
        x = self.relu(self.fn1(x))
        x = self.relu(self.fn2(x))
        
        return self.fn3(x)
