import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, scale_factor):
        super(Generator, self).__init__()
        self.scale_factor = scale_factor

        # Define the upsampling layer
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bicubic', align_corners=True)

        # Define the main network
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4)
        )

    def forward(self, x):
        x = self.upsample(x)
        return self.main(x)

# To be used in import statement
Net = Generator
