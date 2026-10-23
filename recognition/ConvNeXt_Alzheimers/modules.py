"""Small CNN baseline for the COMP3710 ADNI classification investigation.

This is the simpler comparison model, NOT the Hard-Difficulty ConvNeXt model.
Input: preprocessed [B, 1, 224, 224] MRI JPEG slices. Output: two raw logits,
ordered NC=0, AD=1. Use cross-entropy on logits, not on softmax probabilities.

Four Conv/BatchNorm/ReLU blocks downsample spatially. A 4x4 pooled feature map
retains coarse anatomical location before a small fully connected classifier.
The network uses only PyTorch (no NumPy, torchvision, or external model package).

API reference: https://docs.pytorch.org/docs/stable/nn.html
Development assistance: ChatGPT drafted code/tests. The student must review,
validate, explain, and disclose this assistance in the assessment report.
"""
from __future__ import annotations

import torch
from torch import nn


class BaselineCNN(nn.Module):
    """A compact CNN trained from scratch; no downloaded/pretrained weights."""

    def __init__(self, dropout: float = 0.25) -> None:
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        blocks: list[nn.Module] = []
        channels = (1, 16, 32, 64, 128)
        for incoming, outgoing in zip(channels[:-1], channels[1:]):
            blocks.extend([
                nn.Conv2d(incoming, outgoing, kernel_size=3, stride=2,
                          padding=1, bias=False),
                nn.BatchNorm2d(outgoing),
                nn.ReLU(inplace=True),
            ])
        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            # Fixed pooling maps 14x14 to 4x4. Unlike CUDA adaptive-pool
            # backward, this supports the strict deterministic-mode check.
            nn.AvgPool2d(kernel_size=5, stride=3),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, 2),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1:] != (1, 224, 224):
            raise ValueError("Expected image tensor shape [B, 1, 224, 224].")
        return self.classifier(self.features(images))


def parameter_count(model: nn.Module) -> int:
    """Count learnable parameters, not BatchNorm running-statistic buffers."""
    return sum(parameter.numel() for parameter in model.parameters()
               if parameter.requires_grad)
