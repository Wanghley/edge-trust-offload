#!/usr/bin/env python3
"""
build_heavy_model.py
====================
Generates the "Heavy" 1D-CNN gesture classification model specified in the
requirements for the Jetson Nano Orin.

Architecture:
  Input (1024, n_channels)
  -> Conv1D(64, k=7) + BN + ReLU + MaxPool(4)
  -> Conv1D(128, k=5) + BN + ReLU + MaxPool(4)
  -> Conv1D(256, k=3) + BN + ReLU + MaxPool(4)
  -> Conv1D(256, k=3) + BN + ReLU + GlobalAvgPool
  -> Dense(256) + ReLU + Dropout(0.4)
  -> Dense(128) + ReLU + Dropout(0.3)
  -> Dense(6) + Softmax

Output:
  models/gesture_heavy.onnx
"""

import os
import pathlib
import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Constants
WINDOW_SIZE = 1024
N_CHANNELS = 8  # Default case
N_CLASSES = 6
MODEL_DIR = pathlib.Path(__file__).parent.parent / "models"
OUTPUT_PATH = MODEL_DIR / "gesture_heavy.onnx"

class HeavyGestureModel(nn.Module):
    def __init__(self, n_channels=8, n_classes=6):
        super(HeavyGestureModel, self).__init__()
        
        # Convolutional layers
        # Input shape: (Batch, Channels, WindowSize)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(n_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            
            # Block 2
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),
            
            # Block 3
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(4),
            
            # Block 4 (Final extraction)
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # Global Average Pool
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # Transpose input for Conv1d: (Batch, WindowSize, Channels) -> (Batch, Channels, WindowSize)
        x = x.transpose(1, 2)
        x = self.features(x)
        x = self.classifier(x)
        return x

def build_and_export():
    if not HAS_TORCH:
        print("Error: torch not found. Please install it: pip install torch")
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Building 'Heavy' gesture model with {N_CHANNELS} channels...")
    model = HeavyGestureModel(n_channels=N_CHANNELS, n_classes=N_CLASSES)
    model.eval()

    # Create dummy input: (Batch, WindowSize, Channels)
    dummy_input = torch.randn(1, WINDOW_SIZE, N_CHANNELS)
    
    print(f"Exporting to ONNX: {OUTPUT_PATH}")
    torch.onnx.export(
        model, 
        dummy_input, 
        str(OUTPUT_PATH),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    print("Success. Run 'trtexec' on your Jetson to convert this to a TensorRT engine:")
    print(f"  trtexec --onnx={OUTPUT_PATH} --saveEngine={MODEL_DIR}/gesture_heavy.trt --fp16")

if __name__ == "__main__":
    build_and_export()
