"""
models.py
Two models, matching Objectives 2 and 3 of the proposal:

1. ConvAutoencoder: trained ONLY on "normal" spectrograms. Reconstruction
   error at inference time becomes the anomaly / failure-risk score, with
   no fault labels needed at training time (Objective 2).

2. FaultClassifier: trained on labeled spectrograms from all classes,
   distinguishing bearing wear / imbalance / misalignment (Objective 3).
"""

import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),   # 64x196 -> 32x98
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),  # 32x98  -> 16x49
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),  # 16x49  -> 8x25
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        # crop/pad to match input size exactly (stride rounding can shift dims by 1)
        out = nn.functional.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return out


class FaultClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
