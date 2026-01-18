import torch.nn as nn
import torch

class get_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    depth_model = get_model()
    inputs = torch.ones((12,8))
    outputs = depth_model(inputs)
    print(outputs.shape)