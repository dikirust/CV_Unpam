from pathlib import Path

BASE_DIR = Path('.')
DATASET_DIR = BASE_DIR / 'datasets'
TRAIN_DIR = DATASET_DIR / 'train'

print(f"Current directory: {BASE_DIR.absolute()}")
print(f"Dataset directory: {DATASET_DIR}")
print(f"Dataset exists: {DATASET_DIR.exists()}")
print(f"Train directory exists: {TRAIN_DIR.exists()}")

if TRAIN_DIR.exists():
    class_dirs = list(TRAIN_DIR.iterdir())
    print(f"\nClasses found: {len(class_dirs)}")
    for d in class_dirs:
        if d.is_dir():
            img_count = len(list(d.glob('*.jpg')))
            print(f"  - {d.name}: {img_count} images")
else:
    print("\nERROR: datasets/train/ directory not found!")
    print("Please ensure dataset is in the correct location")
