"""
Waste Classification using Deep Learning
Perbandingan Arsitektur: Custom CNN vs MobileNetV2
=====================================================
Author: Diki Rustian
Institution: Universitas Pamulang, Indonesia
Email: diki.rstn@gmail.com

Deskripsi:
Script ini membandingkan dua arsitektur CNN untuk klasifikasi jenis sampah:
1. Custom CNN - Arsitektur custom yang dibangun dari awal
2. MobileNetV2 - Arsitektur transfer learning yang lightweight

Langkah-langkah:
1. Load dan preprocess dataset
2. Train Custom CNN
3. Train MobileNetV2
4. Bandingkan metrik performa (accuracy, loss, precision, recall, F1)
5. Generate visualisasi dan laporan
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import json
from datetime import datetime

# ==================== KONFIGURASI ====================
print("="*80)
print("WASTE CLASSIFICATION - CNN vs MobileNetV2 COMPARISON")
print("="*80)

BASE_DIR = Path('.')
DATASET_DIR = BASE_DIR / 'datasets'
OUTPUT_DIR = BASE_DIR / 'output'
MODEL_DIR = OUTPUT_DIR / 'models'
REPORT_DIR = OUTPUT_DIR / 'report'

# Buat direktori output
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

# Konfigurasi training
IMG_SIZE = 64
BATCH_SIZE = 16
EPOCHS = 50  # Lebih kecil untuk training lebih cepat
VALIDATION_SPLIT = 0.2
RANDOM_STATE = 42

CLASS_NAMES = ['foodwaste', 'glass', 'metal', 'paper', 'plastic']
NUM_CLASSES = len(CLASS_NAMES)

print(f"\nKonfigurasi:")
print(f"  - Dataset directory: {DATASET_DIR}")
print(f"  - Output directory: {OUTPUT_DIR}")
print(f"  - Image size: {IMG_SIZE}x{IMG_SIZE}")
print(f"  - Batch size: {BATCH_SIZE}")
print(f"  - Epochs: {EPOCHS}")
print(f"  - Classes: {NUM_CLASSES} - {CLASS_NAMES}")
print(f"  - TensorFlow version: {tf.__version__}")

# ==================== STEP 1: LOAD DATA ====================
print("\n" + "="*80)
print("STEP 1: MEMUAT DATA DARI FOLDER DATASETS")
print("="*80)

def load_images_from_directory(directory_path, img_size=IMG_SIZE):
    """Load semua gambar dari directory"""
    images = []
    labels = []
    
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_path = Path(directory_path) / class_name
        if class_path.exists():
            image_files = list(class_path.glob('*.jpg'))
            print(f"\nMemuat {class_name}...")
            print(f"  - Total file: {len(image_files)}")
            
            for img_file in image_files:
                try:
                    img = cv2.imread(str(img_file))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (img_size, img_size))
                    img = img.astype(np.float32) / 255.0
                    images.append(img)
                    labels.append(class_idx)
                except Exception as e:
                    print(f"  ✗ Error: {img_file} - {e}")
            
            loaded_count = len([l for l in labels if l == class_idx])
            print(f"  ✓ Loaded: {loaded_count} images")
    
    return np.array(images), np.array(labels)

# Load data
print("\nMemuat data training...")
train_dir = DATASET_DIR / 'train'
X_all, y_all = load_images_from_directory(train_dir)

print(f"\n✓ Data loading complete!")
print(f"  - Total samples: {len(X_all)}")
print(f"  - Image shape: {X_all[0].shape}")

# Split data: train (70%), validation (20%), test (10%)
print("\n" + "-"*80)
print("Membagi dataset:")
print("-"*80)

X_train, X_temp, y_train, y_temp = train_test_split(
    X_all, y_all, test_size=0.3, random_state=RANDOM_STATE, stratify=y_all
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.333, random_state=RANDOM_STATE, stratify=y_temp
)

print(f"Training set:   {len(X_train)} samples ({len(X_train)/len(X_all)*100:.1f}%)")
print(f"Validation set: {len(X_val)} samples ({len(X_val)/len(X_all)*100:.1f}%)")
print(f"Test set:       {len(X_test)} samples ({len(X_test)/len(X_all)*100:.1f}%)")

# Print class distribution
print("\n" + "-"*80)
print("Distribusi class dalam setiap set:")
print("-"*80)
for dataset_name, y_data in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
    print(f"\n{dataset_name}:")
    for i, class_name in enumerate(CLASS_NAMES):
        count = np.sum(y_data == i)
        percentage = (count / len(y_data)) * 100
        print(f"  - {class_name:15s}: {count:5d} ({percentage:5.1f}%)")

# Simpan split information
split_info = {
    'total_samples': len(X_all),
    'train_samples': len(X_train),
    'val_samples': len(X_val),
    'test_samples': len(X_test),
    'class_names': CLASS_NAMES,
    'num_classes': NUM_CLASSES,
    'image_size': IMG_SIZE,
    'batch_size': BATCH_SIZE,
    'epochs': EPOCHS
}

with open(OUTPUT_DIR / 'split_info.json', 'w') as f:
    json.dump(split_info, f, indent=2)

print("\n✓ Data loading selesai!")
print(f"  - Informasi tersimpan di: {OUTPUT_DIR / 'split_info.json'}")
