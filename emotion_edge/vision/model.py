"""Compact facial-expression CNN (mini-Xception style: depthwise-separable blocks, global-avg-pool head).

~60k parameters at 48x48 grayscale: runs in ~1-2 ms on a phone-class core, int8 static-quantizable (QDQ).
Pretrained backbones (MobileNetV3-small, 1.5M+ params) are a drop-in alternative via --backbone but need Hub/torchvision weights.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from emotion_edge.labels import FACE_LABELS


def sep(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False), nn.Conv2d(cin, cout, 1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU())


class MiniXception(nn.Module):
    def __init__(self, n_out=len(FACE_LABELS), w=16):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU())
        self.body = nn.Sequential(sep(w, 2 * w), nn.MaxPool2d(2), sep(2 * w, 4 * w), nn.MaxPool2d(2),
                                  sep(4 * w, 8 * w), nn.MaxPool2d(2), sep(8 * w, 8 * w), sep(8 * w, 16 * w))
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Conv2d(16 * w, n_out, 1), nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(self, x):       # x: (B,1,48,48) in [0,1]
        return self.head(self.body(self.stem((x - 0.5) / 0.25)))
