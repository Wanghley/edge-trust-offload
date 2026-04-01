#!/usr/bin/env python3
"""
build_lite_model.py
===================
Generates the "Lite" 1D-CNN gesture classification model specified in the
requirements for the Raspberry Pi (TFLite).

Output:
  models/gesture_cnn.tflite
"""

import sys
import os
import pathlib
import numpy as np

# Add parent directory to path to import client.edge_tasks
parent_dir = str(pathlib.Path(__file__).resolve().parent.parent)
sys.path.insert(0, parent_dir)

from client.edge_tasks import train_gesture_model, WINDOW_SIZE, N_CLASSES, MODEL_DIR

def main():
    n_samples = 300  # 50 samples per class
    n_channels = 8
    
    print(f"Generating {n_samples} synthetic training samples for the Edge baseline...")
    # Use standard normal random values for X and random integers for the classes
    X = np.random.randn(n_samples, WINDOW_SIZE, n_channels).astype(np.float32)
    y = np.random.randint(0, N_CLASSES, size=(n_samples,)).astype(np.int64)
    
    print("Executing offline TFLite offline training...")
    train_gesture_model(
        X=X, 
        y=y, 
        n_channels=n_channels, 
        complexity="normal", 
        output_dir=MODEL_DIR
    )
    
    print("Done! The models/gesture_cnn.tflite file is ready for the Raspberry Pi.")

if __name__ == "__main__":
    main()
