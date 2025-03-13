import numpy as np
import torch
from torch import nn

# This CNN is meant to take in a smaller image, just containing a hand. (128x256)
# The output will be an integer in the range 0-5, how many fingers are held up.
class HandToDigitsCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.convolutional_relu_stack = nn.Sequential(
                # Input: 3 x 128 x 256
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # 64 x 64 x 128

                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # 128 x 32 x 64

                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(256),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # 256 x 16 x 32
                
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(512),
                # 512 x 16 x 32
        )

        self.linear_stack = nn.Sequential(
            nn.Linear(512 * 16 * 32, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 6)
        )

    def forward(self, x):
        x = self.convolutional_relu_stack(x)
        x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits
