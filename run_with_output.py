#!/usr/bin/env python
"""
Test script to run waste classification with visible output
"""
import sys
import os




# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'

print("=" * 80)
print("WASTE CLASSIFICATION SCRIPT - EXECUTION START")
print("=" * 80)
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")
sys.stdout.flush()

# Check TensorFlow GPU
print("\nChecking TensorFlow setup...")
import tensorflow as tf
sys.stdout.flush()

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
print(f"CPU cores: {tf.config.list_physical_devices('CPU')}")
sys.stdout.flush()

print("\nImporting required libraries...")
sys.stdout.flush()

import cv2
import numpy as np
from pathlib import Path
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("✓ All imports successful!")
sys.stdout.flush()

# ==================== KONFIGURASI ====================
print("\n" + "=" * 80)
print("STEP 1: SETUP KONFIGURASI")
print("=" * 80)

BASE_DIR = Path('.')
DATASET_DIR = BASE_DIR / 'datasets'
OUTPUT_DIR = BASE_DIR / 'output'
MODEL_DIR = OUTPUT_DIR / 'models'
REPORT_DIR = OUTPUT_DIR / 'report'

# Buat direktori output
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

IMG_SIZE = 64
BATCH_SIZE = 16
EPOCHS = 50
RANDOM_STATE = 42

CLASS_NAMES = ['foodwaste', 'glass', 'metal', 'paper', 'plastic']
NUM_CLASSES = len(CLASS_NAMES)

print(f"Output directory: {OUTPUT_DIR}")
print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Classes: {NUM_CLASSES} - {CLASS_NAMES}")
print("✓ Configuration done!")
sys.stdout.flush()

# ==================== LOAD DATA ====================
print("\n" + "=" * 80)
print("STEP 2: LOADING DATA")
print("=" * 80)

def load_images_from_directory(directory_path, img_size=IMG_SIZE):
    """Load semua gambar dari directory"""
    images = []
    labels = []
    
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_path = Path(directory_path) / class_name
        if class_path.exists():
            image_files = list(class_path.glob('*.jpg'))
            print(f"\nLoading {class_name}... ({len(image_files)} files)")
            sys.stdout.flush()
            
            count = 0
            for img_file in image_files:
                try:
                    img = cv2.imread(str(img_file))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (img_size, img_size))
                    img = img.astype(np.float32) / 255.0
                    images.append(img)
                    labels.append(class_idx)
                    count += 1
                    if count % 500 == 0:
                        print(f"  Loaded {count} images...", end='\r')
                        sys.stdout.flush()
                except Exception as e:
                    print(f"Error: {img_file}")
            
            loaded_count = len([l for l in labels if l == class_idx])
            print(f"✓ {class_name}: {loaded_count} images loaded")
            sys.stdout.flush()
    
    return np.array(images), np.array(labels)

print("\nLoading images...")
sys.stdout.flush()

train_dir = DATASET_DIR / 'train'
X_all, y_all = load_images_from_directory(train_dir)

print(f"\n✓ Data loaded: {len(X_all)} samples")
print(f"  Image shape: {X_all[0].shape}")
sys.stdout.flush()

# ==================== SPLIT DATA ====================
print("\n" + "-" * 80)
print("Splitting dataset...")
sys.stdout.flush()

X_train, X_temp, y_train, y_temp = train_test_split(
    X_all, y_all, test_size=0.3, random_state=RANDOM_STATE, stratify=y_all
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
print("✓ Data split complete!")
sys.stdout.flush()

# ==================== BUILD MODELS ====================
print("\n" + "=" * 80)
print("STEP 3: BUILDING MODELS")
print("=" * 80)

print("Building Custom CNN...")
sys.stdout.flush()

def create_custom_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ], name='CustomCNN')
    return model

custom_cnn = create_custom_cnn()
custom_cnn.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
print("✓ Custom CNN created")
sys.stdout.flush()

print("Building MobileNetV2...")
sys.stdout.flush()

def create_mobilenetv2(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
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

mobilenet_v2 = create_mobilenetv2()
mobilenet_v2.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
print("✓ MobileNetV2 created")
sys.stdout.flush()

print("\nModel summaries:")
print(f"Custom CNN parameters: {custom_cnn.count_params():,}")
print(f"MobileNetV2 parameters: {mobilenet_v2.count_params():,}")
sys.stdout.flush()

# ==================== TRAIN ====================
print("\n" + "=" * 80)
print("STEP 4: TRAINING MODELS")
print("=" * 80)

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

print("\nTraining Custom CNN...")
sys.stdout.flush()

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

print(f"\n✓ Custom CNN trained in {training_time_cnn:.2f} seconds")
sys.stdout.flush()

print("\nTraining MobileNetV2...")
sys.stdout.flush()

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

print(f"\n✓ MobileNetV2 trained in {training_time_mb:.2f} seconds")
sys.stdout.flush()

# ==================== EVALUATE ====================
print("\n" + "=" * 80)
print("STEP 5: EVALUATING MODELS")
print("=" * 80)

def evaluate_model(model, X_test, y_test, model_name):
    print(f"\nEvaluating {model_name}...")
    sys.stdout.flush()
    
    predictions = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    y_true = y_test
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
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
        'true_labels': y_true
    }

results_custom_cnn = evaluate_model(custom_cnn, X_test, y_test, "Custom CNN")
results_mobilenet = evaluate_model(mobilenet_v2, X_test, y_test, "MobileNetV2")

print("\n" + "=" * 80)
print("CUSTOM CNN - TEST RESULTS")
print("=" * 80)
print(f"Accuracy:  {results_custom_cnn['accuracy']:.4f}")
print(f"Precision: {results_custom_cnn['precision']:.4f}")
print(f"Recall:    {results_custom_cnn['recall']:.4f}")
print(f"F1-Score:  {results_custom_cnn['f1_score']:.4f}")
sys.stdout.flush()

print("\n" + "=" * 80)
print("MOBILENETV2 - TEST RESULTS")
print("=" * 80)
print(f"Accuracy:  {results_mobilenet['accuracy']:.4f}")
print(f"Precision: {results_mobilenet['precision']:.4f}")
print(f"Recall:    {results_mobilenet['recall']:.4f}")
print(f"F1-Score:  {results_mobilenet['f1_score']:.4f}")
sys.stdout.flush()

# ==================== SAVE RESULTS ====================
print("\n" + "=" * 80)
print("STEP 6: SAVING RESULTS")
print("=" * 80)

print("Saving models...")
sys.stdout.flush()
custom_cnn.save(str(MODEL_DIR / 'custom_cnn.h5'))
mobilenet_v2.save(str(MODEL_DIR / 'mobilenetv2.h5'))
print("✓ Models saved")
sys.stdout.flush()

print("Saving evaluation results...")
sys.stdout.flush()

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
    },
    'training_times': {
        'custom_cnn': training_time_cnn,
        'mobilenetv2': training_time_mb
    }
}

with open(OUTPUT_DIR / 'evaluation_results.json', 'w') as f:
    json.dump(evaluation_results, f, indent=2)
print("✓ Results saved to evaluation_results.json")
sys.stdout.flush()

# ==================== VISUALIZATIONS ====================
print("\n" + "=" * 80)
print("STEP 7: GENERATING VISUALIZATIONS")
print("=" * 80)

print("Generating comparison charts...")
sys.stdout.flush()

# Metrics comparison
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(4)
width = 0.35

metrics_custom = [results_custom_cnn['accuracy'], results_custom_cnn['precision'],
                  results_custom_cnn['recall'], results_custom_cnn['f1_score']]
metrics_mobile = [results_mobilenet['accuracy'], results_mobilenet['precision'],
                  results_mobilenet['recall'], results_mobilenet['f1_score']]

bars1 = ax.bar(x - width/2, metrics_custom, width, label='Custom CNN', color='#3498db')
bars2 = ax.bar(x + width/2, metrics_mobile, width, label='MobileNetV2', color='#2ecc71')

ax.set_ylabel('Score')
ax.set_title('Model Metrics Comparison', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1-Score'])
ax.legend()
ax.set_ylim([0, 1.0])
ax.grid(True, alpha=0.3, axis='y')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(str(REPORT_DIR / 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ metrics_comparison.png saved")
sys.stdout.flush()

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(results_custom_cnn['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0])
axes[0].set_title('Custom CNN - Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('True')

sns.heatmap(results_mobilenet['confusion_matrix'], annot=True, fmt='d', cmap='Greens',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[1])
axes[1].set_title('MobileNetV2 - Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('True')

plt.tight_layout()
plt.savefig(str(REPORT_DIR / 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ confusion_matrices.png saved")
sys.stdout.flush()

# Per-class accuracy
class_names_list = list(results_custom_cnn['per_class'].keys())
custom_acc = [results_custom_cnn['per_class'][c]['accuracy'] for c in class_names_list]
mobile_acc = [results_mobilenet['per_class'][c]['accuracy'] for c in class_names_list]

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(class_names_list))
width = 0.35

bars1 = ax.bar(x - width/2, custom_acc, width, label='Custom CNN', color='#3498db')
bars2 = ax.bar(x + width/2, mobile_acc, width, label='MobileNetV2', color='#2ecc71')

ax.set_ylabel('Accuracy')
ax.set_title('Per-Class Accuracy Comparison', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(class_names_list, rotation=45)
ax.legend()
ax.set_ylim([0, 1.0])
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(str(REPORT_DIR / 'per_class_accuracy.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ per_class_accuracy.png saved")
sys.stdout.flush()

# ==================== COMPLETION ====================
print("\n" + "=" * 80)
print("EXECUTION COMPLETED SUCCESSFULLY!")
print("=" * 80)
print(f"\nResults saved in: {OUTPUT_DIR}")
print(f"  - Models: {MODEL_DIR}")
print(f"  - Visualizations: {REPORT_DIR}")
print("\nGenerated files:")
print(f"  ✓ custom_cnn.h5")
print(f"  ✓ mobilenetv2.h5")
print(f"  ✓ evaluation_results.json")
print(f"  ✓ metrics_comparison.png")
print(f"  ✓ confusion_matrices.png")
print(f"  ✓ per_class_accuracy.png")
print("\n" + "=" * 80)
print("Phase 1 Complete - Ready for Phase 2 (Report Generation)")
print("=" * 80)
sys.stdout.flush()
