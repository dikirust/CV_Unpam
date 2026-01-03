"""
Waste Classification using Deep Learning
Perbandingan Arsitektur: Custom CNN vs MobileNetV2
=====================================================
Author: Diki Rustian
Institution: Universitas Pamulang, Indonesia
Email: diki.rstn@gmail.com

Langkah-langkah Eksekusi:
1. Load dan preprocess dataset
2. Definisikan 2 arsitektur: Custom CNN dan MobileNetV2
3. Train kedua model
4. Evaluasi dan bandingkan performa
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
import warnings
warnings.filterwarnings('ignore')

# ==================== KONFIGURASI ====================
print("="*80)
print("WASTE CLASSIFICATION - CNN vs MobileNetV2 COMPARISON")
print("="*80)
print(f"TensorFlow version: {tf.__version__}")
# Keras is part of TensorFlow in newer versions

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
EPOCHS = 50
VALIDATION_SPLIT = 0.2
RANDOM_STATE = 42

CLASS_NAMES = ['foodwaste', 'glass', 'metal', 'paper', 'plastic']
NUM_CLASSES = len(CLASS_NAMES)

print(f"\nKonfigurasi Training:")
print(f"  - Image size: {IMG_SIZE}x{IMG_SIZE}")
print(f"  - Batch size: {BATCH_SIZE}")
print(f"  - Epochs: {EPOCHS}")
print(f"  - Classes: {NUM_CLASSES} - {CLASS_NAMES}")

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

# Split data: train (70%), validation (15%), test (15%)
print("\n" + "-"*80)
print("Membagi dataset...")
print("-"*80)

X_train, X_temp, y_train, y_temp = train_test_split(
    X_all, y_all, test_size=0.3, random_state=RANDOM_STATE, stratify=y_all
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
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

# ==================== STEP 2: DEFINISI ARSITEKTUR ====================
print("\n" + "="*80)
print("STEP 2: MENDEFINISIKAN ARSITEKTUR MODEL")
print("="*80)

def create_custom_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    """
    Custom CNN Architecture
    - 3 Convolutional blocks
    - Filter progression: 32 -> 64 -> 128
    - Dropout dan BatchNormalization untuk regularization
    """
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Global pooling
        layers.GlobalAveragePooling2D(),
        
        # Dense layers
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Output
        layers.Dense(num_classes, activation='softmax')
    ], name='CustomCNN')
    
    return model


def create_mobilenetv2(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    """
    MobileNetV2 Architecture (Lightweight)
    - Transfer learning dari pretrained ImageNet
    - Efficient depthwise separable convolutions
    """
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    base_model.trainable = False
    
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ], name='MobileNetV2')
    
    return model


# Buat kedua model
print("\nMembuat Custom CNN...")
custom_cnn = create_custom_cnn()
print("✓ Custom CNN created")

print("Membuat MobileNetV2...")
mobilenet_v2 = create_mobilenetv2()
print("✓ MobileNetV2 created")

# Compile kedua model
print("\nMeng-compile model...")

custom_cnn.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
print("✓ Custom CNN compiled")

mobilenet_v2.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
print("✓ MobileNetV2 compiled")

# Print model summaries
print("\n" + "="*80)
print("MODEL 1: CUSTOM CNN - ARCHITECTURE SUMMARY")
print("="*80)
custom_cnn.summary()

print("\n" + "="*80)
print("MODEL 2: MOBILENETV2 - ARCHITECTURE SUMMARY")
print("="*80)
mobilenet_v2.summary()

# Perbandingan parameter
print("\n" + "="*80)
print("PERBANDINGAN PARAMETER MODEL")
print("="*80)
custom_params = custom_cnn.count_params()
mobilenet_params = mobilenet_v2.count_params()

print(f"\nCustom CNN:")
print(f"  - Total Parameters: {custom_params:,}")

print(f"\nMobileNetV2:")
print(f"  - Total Parameters: {mobilenet_params:,}")
print(f"  - Ukuran lebih kecil {(1 - mobilenet_params/custom_params)*100:.1f}% dari Custom CNN")

# Simpan parameter info
models_info = {
    'custom_cnn': {
        'name': 'Custom CNN',
        'parameters': int(custom_params),
        'description': '3 Convolutional blocks dengan BatchNormalization dan Dropout'
    },
    'mobilenetv2': {
        'name': 'MobileNetV2',
        'parameters': int(mobilenet_params),
        'description': 'Transfer learning dari ImageNet dengan depthwise separable convolutions'
    }
}

with open(OUTPUT_DIR / 'models_info.json', 'w') as f:
    json.dump(models_info, f, indent=2)

print("\n✓ Model definitions selesai!")

# ==================== STEP 3: TRAINING MODEL ====================
print("\n" + "="*80)
print("STEP 3: MELATIH KEDUA MODEL")
print("="*80)

# Setup callbacks
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-8,
    verbose=1
)

# Training Custom CNN
print("\n" + "-"*80)
print("Melatih Custom CNN...")
print("-"*80)

start_time_cnn = datetime.now()

history_custom_cnn = custom_cnn.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

end_time_cnn = datetime.now()
training_time_cnn = (end_time_cnn - start_time_cnn).total_seconds()

print(f"\n✓ Training Custom CNN selesai!")
print(f"  - Training time: {training_time_cnn:.2f} seconds")
print(f"  - Epochs trained: {len(history_custom_cnn.history['loss'])}")

# Training MobileNetV2
print("\n" + "-"*80)
print("Melatih MobileNetV2...")
print("-"*80)

start_time_mb = datetime.now()

history_mobilenet = mobilenet_v2.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

end_time_mb = datetime.now()
training_time_mb = (end_time_mb - start_time_mb).total_seconds()

print(f"\n✓ Training MobileNetV2 selesai!")
print(f"  - Training time: {training_time_mb:.2f} seconds")
print(f"  - Epochs trained: {len(history_mobilenet.history['loss'])}")

# Simpan training history
with open(OUTPUT_DIR / 'history_custom_cnn.pkl', 'wb') as f:
    pickle.dump(history_custom_cnn.history, f)

with open(OUTPUT_DIR / 'history_mobilenetv2.pkl', 'wb') as f:
    pickle.dump(history_mobilenet.history, f)

print("\n✓ Training selesai dan history tersimpan!")

# ==================== STEP 4: VISUALISASI TRAINING ====================
print("\n" + "="*80)
print("STEP 4: VISUALISASI TRAINING CURVES")
print("="*80)

# Plot training curves - Custom CNN
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history_custom_cnn.history['accuracy'], label='Training', marker='o')
axes[0].plot(history_custom_cnn.history['val_accuracy'], label='Validation', marker='s')
axes[0].set_title('Custom CNN - Accuracy', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history_custom_cnn.history['loss'], label='Training', marker='o')
axes[1].plot(history_custom_cnn.history['val_loss'], label='Validation', marker='s')
axes[1].set_title('Custom CNN - Loss', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(str(REPORT_DIR / '01_custom_cnn_training.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 01_custom_cnn_training.png")

# Plot training curves - MobileNetV2
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history_mobilenet.history['accuracy'], label='Training', marker='o')
axes[0].plot(history_mobilenet.history['val_accuracy'], label='Validation', marker='s')
axes[0].set_title('MobileNetV2 - Accuracy', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history_mobilenet.history['loss'], label='Training', marker='o')
axes[1].plot(history_mobilenet.history['val_loss'], label='Validation', marker='s')
axes[1].set_title('MobileNetV2 - Loss', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(str(REPORT_DIR / '02_mobilenetv2_training.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 02_mobilenetv2_training.png")

# Plot perbandingan training curves
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Accuracy comparison
axes[0, 0].plot(history_custom_cnn.history['accuracy'], label='Custom CNN', marker='o', linewidth=2)
axes[0, 0].plot(history_mobilenet.history['accuracy'], label='MobileNetV2', marker='s', linewidth=2)
axes[0, 0].set_title('Training Accuracy Comparison', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Validation accuracy comparison
axes[0, 1].plot(history_custom_cnn.history['val_accuracy'], label='Custom CNN', marker='o', linewidth=2)
axes[0, 1].plot(history_mobilenet.history['val_accuracy'], label='MobileNetV2', marker='s', linewidth=2)
axes[0, 1].set_title('Validation Accuracy Comparison', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Loss comparison
axes[1, 0].plot(history_custom_cnn.history['loss'], label='Custom CNN', marker='o', linewidth=2)
axes[1, 0].plot(history_mobilenet.history['loss'], label='MobileNetV2', marker='s', linewidth=2)
axes[1, 0].set_title('Training Loss Comparison', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Validation loss comparison
axes[1, 1].plot(history_custom_cnn.history['val_loss'], label='Custom CNN', marker='o', linewidth=2)
axes[1, 1].plot(history_mobilenet.history['val_loss'], label='MobileNetV2', marker='s', linewidth=2)
axes[1, 1].set_title('Validation Loss Comparison', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Loss')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(str(REPORT_DIR / '03_training_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 03_training_comparison.png")

# Simpan training info
training_info = {
    'custom_cnn': {
        'training_time_seconds': training_time_cnn,
        'epochs_trained': len(history_custom_cnn.history['loss']),
        'final_accuracy': float(history_custom_cnn.history['accuracy'][-1]),
        'final_val_accuracy': float(history_custom_cnn.history['val_accuracy'][-1]),
        'final_loss': float(history_custom_cnn.history['loss'][-1]),
        'final_val_loss': float(history_custom_cnn.history['val_loss'][-1])
    },
    'mobilenetv2': {
        'training_time_seconds': training_time_mb,
        'epochs_trained': len(history_mobilenet.history['loss']),
        'final_accuracy': float(history_mobilenet.history['accuracy'][-1]),
        'final_val_accuracy': float(history_mobilenet.history['val_accuracy'][-1]),
        'final_loss': float(history_mobilenet.history['loss'][-1]),
        'final_val_loss': float(history_mobilenet.history['val_loss'][-1])
    }
}

with open(OUTPUT_DIR / 'training_info.json', 'w') as f:
    json.dump(training_info, f, indent=2)

print("\n✓ Training visualization selesai!")

# ==================== STEP 5: EVALUASI DI TEST SET ====================
print("\n" + "="*80)
print("STEP 5: EVALUASI MODEL DI TEST SET")
print("="*80)

def evaluate_model(model, X_test, y_test, model_name, batch_size=BATCH_SIZE):
    """Evaluasi model dan hitung semua metrik performa"""
    print(f"\nMengevaluasi {model_name}...")
    
    # Prediksi
    predictions = model.predict(X_test, batch_size=batch_size, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    y_true = y_test
    
    # Hitung metrik
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    # Per-class metrics
    per_class = {}
    for i, class_name in enumerate(CLASS_NAMES):
        class_mask = y_true == i
        if np.sum(class_mask) > 0:
            class_accuracy = np.sum((y_pred == i) & class_mask) / np.sum(class_mask)
            per_class[class_name] = {
                'accuracy': float(class_accuracy),
                'samples': int(np.sum(class_mask))
            }
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'per_class': per_class,
        'predictions': y_pred,
        'true_labels': y_true,
        'class_report': classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    }

# Evaluasi Custom CNN
results_custom_cnn = evaluate_model(custom_cnn, X_test, y_test, "Custom CNN")

# Evaluasi MobileNetV2
results_mobilenet = evaluate_model(mobilenet_v2, X_test, y_test, "MobileNetV2")

# Print hasil evaluasi Custom CNN
print("\n" + "-"*80)
print("CUSTOM CNN - TEST SET PERFORMANCE")
print("-"*80)
print(f"Accuracy:  {results_custom_cnn['accuracy']:.4f} ({results_custom_cnn['accuracy']*100:.2f}%)")
print(f"Precision: {results_custom_cnn['precision']:.4f}")
print(f"Recall:    {results_custom_cnn['recall']:.4f}")
print(f"F1-Score:  {results_custom_cnn['f1_score']:.4f}")
print("\nPer-Class Accuracy:")
for class_name, metrics in results_custom_cnn['per_class'].items():
    print(f"  - {class_name:15s}: {metrics['accuracy']:.4f} ({metrics['samples']} samples)")

# Print hasil evaluasi MobileNetV2
print("\n" + "-"*80)
print("MOBILENETV2 - TEST SET PERFORMANCE")
print("-"*80)
print(f"Accuracy:  {results_mobilenet['accuracy']:.4f} ({results_mobilenet['accuracy']*100:.2f}%)")
print(f"Precision: {results_mobilenet['precision']:.4f}")
print(f"Recall:    {results_mobilenet['recall']:.4f}")
print(f"F1-Score:  {results_mobilenet['f1_score']:.4f}")
print("\nPer-Class Accuracy:")
for class_name, metrics in results_mobilenet['per_class'].items():
    print(f"  - {class_name:15s}: {metrics['accuracy']:.4f} ({metrics['samples']} samples)")

# ==================== STEP 6: VISUALISASI HASIL EVALUASI ====================
print("\n" + "="*80)
print("STEP 6: VISUALISASI HASIL EVALUASI")
print("="*80)

# Confusion Matrix - Custom CNN
plt.figure(figsize=(8, 6))
sns.heatmap(results_custom_cnn['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Custom CNN - Confusion Matrix (Test Set)', fontsize=12, fontweight='bold')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(str(REPORT_DIR / '04_cm_custom_cnn.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 04_cm_custom_cnn.png")

# Confusion Matrix - MobileNetV2
plt.figure(figsize=(8, 6))
sns.heatmap(results_mobilenet['confusion_matrix'], annot=True, fmt='d', cmap='Greens',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('MobileNetV2 - Confusion Matrix (Test Set)', fontsize=12, fontweight='bold')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(str(REPORT_DIR / '05_cm_mobilenetv2.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 05_cm_mobilenetv2.png")

# Performance Metrics Comparison
metrics_comparison = {
    'Custom CNN': [
        results_custom_cnn['accuracy'],
        results_custom_cnn['precision'],
        results_custom_cnn['recall'],
        results_custom_cnn['f1_score']
    ],
    'MobileNetV2': [
        results_mobilenet['accuracy'],
        results_mobilenet['precision'],
        results_mobilenet['recall'],
        results_mobilenet['f1_score']
    ]
}

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(['Accuracy', 'Precision', 'Recall', 'F1-Score']))
width = 0.35

bars1 = ax.bar(x - width/2, metrics_comparison['Custom CNN'], width, label='Custom CNN', color='#3498db')
bars2 = ax.bar(x + width/2, metrics_comparison['MobileNetV2'], width, label='MobileNetV2', color='#2ecc71')

ax.set_ylabel('Score', fontsize=11)
ax.set_title('Perbandingan Metrik Performa - Test Set', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1-Score'])
ax.legend()
ax.set_ylim([0, 1.0])
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(str(REPORT_DIR / '06_metrics_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 06_metrics_comparison.png")

# Per-Class Accuracy Comparison
class_names_list = list(results_custom_cnn['per_class'].keys())
custom_cnn_acc = [results_custom_cnn['per_class'][c]['accuracy'] for c in class_names_list]
mobilenet_acc = [results_mobilenet['per_class'][c]['accuracy'] for c in class_names_list]

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(class_names_list))
width = 0.35

bars1 = ax.bar(x - width/2, custom_cnn_acc, width, label='Custom CNN', color='#3498db')
bars2 = ax.bar(x + width/2, mobilenet_acc, width, label='MobileNetV2', color='#2ecc71')

ax.set_ylabel('Accuracy', fontsize=11)
ax.set_title('Per-Class Accuracy Comparison - Test Set', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(class_names_list, rotation=45)
ax.legend()
ax.set_ylim([0, 1.0])
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(str(REPORT_DIR / '07_per_class_accuracy.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 07_per_class_accuracy.png")

# ==================== STEP 7: RINGKASAN PERBANDINGAN ====================
print("\n" + "="*80)
print("STEP 7: RINGKASAN PERBANDINGAN")
print("="*80)

summary_fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Accuracy comparison
models = ['Custom CNN', 'MobileNetV2']
accuracies = [results_custom_cnn['accuracy'], results_mobilenet['accuracy']]
colors = ['#3498db', '#2ecc71']

axes[0, 0].bar(models, accuracies, color=colors, alpha=0.7)
axes[0, 0].set_ylabel('Accuracy', fontsize=11)
axes[0, 0].set_title('Test Accuracy Comparison', fontsize=12, fontweight='bold')
axes[0, 0].set_ylim([0, 1.0])
axes[0, 0].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(accuracies):
    axes[0, 0].text(i, v + 0.02, f'{v:.4f}', ha='center', fontsize=10, fontweight='bold')

# Model parameters
params = [custom_params, mobilenet_params]
axes[0, 1].bar(models, params, color=colors, alpha=0.7)
axes[0, 1].set_ylabel('Number of Parameters', fontsize=11)
axes[0, 1].set_title('Model Parameters', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(params):
    axes[0, 1].text(i, v + v*0.02, f'{v:,.0f}', ha='center', fontsize=10, fontweight='bold')

# Training time
training_times = [training_time_cnn, training_time_mb]
axes[1, 0].bar(models, training_times, color=colors, alpha=0.7)
axes[1, 0].set_ylabel('Training Time (seconds)', fontsize=11)
axes[1, 0].set_title('Training Time', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(training_times):
    axes[1, 0].text(i, v + v*0.02, f'{v:.2f}s', ha='center', fontsize=10, fontweight='bold')

# F1-Score comparison
f1_scores = [results_custom_cnn['f1_score'], results_mobilenet['f1_score']]
axes[1, 1].bar(models, f1_scores, color=colors, alpha=0.7)
axes[1, 1].set_ylabel('F1-Score', fontsize=11)
axes[1, 1].set_title('F1-Score Comparison', fontsize=12, fontweight='bold')
axes[1, 1].set_ylim([0, 1.0])
axes[1, 1].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(f1_scores):
    axes[1, 1].text(i, v + 0.02, f'{v:.4f}', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(str(REPORT_DIR / '08_summary_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 08_summary_comparison.png")

# Simpan hasil evaluasi
evaluation_results = {
    'custom_cnn': {
        'accuracy': float(results_custom_cnn['accuracy']),
        'precision': float(results_custom_cnn['precision']),
        'recall': float(results_custom_cnn['recall']),
        'f1_score': float(results_custom_cnn['f1_score']),
        'per_class': results_custom_cnn['per_class']
    },
    'mobilenetv2': {
        'accuracy': float(results_mobilenet['accuracy']),
        'precision': float(results_mobilenet['precision']),
        'recall': float(results_mobilenet['recall']),
        'f1_score': float(results_mobilenet['f1_score']),
        'per_class': results_mobilenet['per_class']
    }
}

with open(OUTPUT_DIR / 'evaluation_results.json', 'w') as f:
    json.dump(evaluation_results, f, indent=2)

# Simpan model
print("\nMenyimpan model...")
custom_cnn.save(str(MODEL_DIR / 'custom_cnn.h5'))
mobilenet_v2.save(str(MODEL_DIR / 'mobilenetv2.h5'))
print("✓ Model tersimpan di folder models/")

# ==================== SELESAI ====================
print("\n" + "="*80)
print("PROSES SELESAI!")
print("="*80)
print(f"\nOutput files tersimpan di: {OUTPUT_DIR}")
print(f"  - Models: {MODEL_DIR}")
print(f"  - Reports & Visualisasi: {REPORT_DIR}")
print(f"\nFile yang dihasilkan:")
print(f"  - JSON files: split_info, models_info, training_info, evaluation_results")
print(f"  - Images: 01-08 (visualisasi)")
print(f"  - Models: custom_cnn.h5, mobilenetv2.h5")

print("\nSiap untuk generate report step 2!")
