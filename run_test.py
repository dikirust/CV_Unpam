"""
Quick test - first steps of waste classification
"""
import os
import sys

print("Starting waste classification pipeline...")
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")

# Test imports
print("\nTesting imports...")
try:
    import tensorflow as tf
    print(f"✓ TensorFlow {tf.__version__}")
except Exception as e:
    print(f"✗ TensorFlow error: {e}")
    sys.exit(1)

try:
    from tensorflow import keras
    print(f"✓ Keras (from TensorFlow)")
except Exception as e:
    print(f"✗ Keras error: {e}")
    sys.exit(1)

try:
    import cv2
    print(f"✓ OpenCV available")
except Exception as e:
    print(f"✗ OpenCV error: {e}")
    sys.exit(1)

try:
    import numpy as np
    print(f"✓ NumPy {np.__version__}")
except Exception as e:
    print(f"✗ NumPy error: {e}")
    sys.exit(1)

try:
    from sklearn.metrics import accuracy_score
    print(f"✓ Scikit-learn available")
except Exception as e:
    print(f"✗ Scikit-learn error: {e}")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    print(f"✓ Matplotlib available")
except Exception as e:
    print(f"✗ Matplotlib error: {e}")
    sys.exit(1)

print("\n✓ All imports successful!")
print("\nNow running the main script...")
print("="*80)

# Now run the actual main script
exec(open('waste_classification_main.py').read())
