import numpy as np
import torch
from torch import nn

# This CNN is meant to take in an image, in which somewhere there will be a hand.
# The output is 4 integers, representing the left, top, right, and bottom of a box.
# This box should surround the hand, simplifying the classification task for another CNN.
class ImageToHandBoxCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.convolutional_relu_stack = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(256),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(512),
                nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.linear_stack = nn.Sequential(
            nn.Linear(512 * 32 * 32, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 4)
        )

    def forward(self, x):
        x = self.convolutional_relu_stack(x)
        x = self.flatten(x)
        logits = self.linear_stack(x)

        x1, y1, x2, y2 = logits[:, 0], logits[:, 1], logits[:, 2], logits[:, 3]
        #print(x1, y1, x2, y2)

        #x1 = torch.sigmoid(x1) * 512
        #y1 = torch.sigmoid(y1) * 512
        x2 = x1 + torch.exp(x2)
        y2 = y1 + torch.exp(y2)

        return torch.stack([x1, y1, x2, y2], dim=1)
