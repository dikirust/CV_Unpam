
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
    - Ideal untuk device dengan resource terbatas
    """
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model untuk transfer learning
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
print(f"  - Size Reduction: {(1 - mobilenet_params/custom_params)*100:.1f}%")

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
print(f"  - Informasi tersimpan di: {OUTPUT_DIR / 'models_info.json'}")
