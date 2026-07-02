import torch
import torch.nn as nn

class NNEU(nn.Module):

    in_features = 40960

    def __init__(self):
        super().__init__()
        
        self.feature_extractor = nn.Linear(in_features=self.in_features, out_features=256)
        self.fn1 = nn.Linear(2 * 256, 32)
        self.fn2 = nn.Linear(32, 32)
        self.fn3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, white_features, black_features, side): 
        w = self.feature_extractor(white_features)
        b = self.feature_extractor(black_features)
        x = torch.cat([w, b], dim=1) if side else torch.cat([b, w], dim=1)
        x = self.relu(x)
        x = self.relu(self.fn1(x))
        x = self.relu(self.fn2(x))
        
        return self.fn3(x)
