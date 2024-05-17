import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, scale_factor):
        super(Generator, self).__init__()
        self.scale_factor = scale_factor
        
        # Define a simple model for illustration
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4)
        )

    def forward(self, x):
        return self.main(x)

# To be used in import statement
Net = Generator
