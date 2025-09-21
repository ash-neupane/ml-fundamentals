import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, device = "cpu"):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features = 28*28, out_features = 512, bias = True, device = device),
            nn.ReLU(),
            nn.Linear(in_features = 512, out_features = 256, bias = True, device = device),
            nn.ReLU(), 
            nn.Linear(in_features = 256, out_features = 10, bias = True, device = device)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)
    

