
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
print(f"  - Informasi tersimpan di: {OUTPUT_DIR / 'training_info.json'}")
