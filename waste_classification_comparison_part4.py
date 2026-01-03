
# ==================== STEP 5: EVALUASI DI TEST SET ====================
print("\n" + "="*80)
print("STEP 5: EVALUASI MODEL DI TEST SET")
print("="*80)

def evaluate_model(model, X_test, y_test, model_name, batch_size=BATCH_SIZE):
    """
    Evaluasi model dan hitung semua metrik performa
    """
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

print("\n✓ Evaluasi selesai!")
print(f"  - Hasil tersimpan di: {OUTPUT_DIR / 'evaluation_results.json'}")

# Simpan model
print("\nMenyimpan model...")
custom_cnn.save(str(MODEL_DIR / 'custom_cnn.h5'))
mobilenet_v2.save(str(MODEL_DIR / 'mobilenetv2.h5'))
print("✓ Model tersimpan di folder models/")
