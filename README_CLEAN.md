# Waste Classification CNN & MobileNetV2 Comparison

Perbandingan Arsitektur Deep Learning untuk Klasifikasi Sampah: Analisis Custom CNN dan MobileNetV2

## 📁 Struktur Project (Clean)

```
CV_Unpam/
├── datasets/
│   ├── train/          # 70% training data (5,880 images)
│   ├── valid/          # 15% validation data (1,260 images)
│   └── test/           # 15% test data (1,260 images)
│
├── models/
│   ├── custom_cnn.h5          # Custom CNN model (2.0 MB)
│   ├── mobilenetv2.h5         # MobileNetV2 model (13.7 MB)
│   └── waste_classification_model.h5  # Backup model
│
├── output/
│   ├── report_jutif_final.docx        # Final JUTIF journal article ✓
│   ├── evaluation_results.json        # Model metrics & results
│   └── [visualizations & outputs]
│
├── report/
│   ├── metrics_comparison.png  # Performance metrics comparison
│   └── [other PNG visualizations]
│
├── waste_classification_main.py  # Main training script
├── generate_report_clean.py      # Clean report generator
├── waste_classification_cnn.ipynb # Jupyter notebook
├── requirements.txt              # Python dependencies
└── README.md                      # Project documentation
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Models (if needed)

```bash
python waste_classification_main.py
```

### 3. Generate Report

```bash
python generate_report_clean.py
```

## 📊 Model Performance

| Metric        | Custom CNN | MobileNetV2                |
| ------------- | ---------- | -------------------------- |
| Accuracy      | 90.16%     | **93.65%** ✓               |
| Precision     | 90.54%     | **93.70%**                 |
| Recall        | 90.16%     | **93.65%**                 |
| F1-Score      | 90.14%     | **93.67%**                 |
| Training Time | 451 sec    | **231 sec** (1.95× faster) |
| Parameters    | 310,405    | 2,259,297                  |

## 📄 Output Files

- **report_jutif_final.docx** - Publication-ready journal article

  - JUTIF-compliant format
  - Bahasa Indonesia + English technical terms
  - 27 IEEE-format references
  - 3 embedded visualizations
  - Tab-indented paragraphs

- **evaluation_results.json** - Complete model metrics
- **PNG visualizations** - Performance charts and confusion matrices

## 📝 Files Removed (Cleanup)

Files not related to main logic were removed:

- ❌ 21 documentation/info TXT files
- ❌ 5 intermediate report generation scripts
- ❌ 7 test/experimental scripts
- ❌ template/ folder

**Kept:** Only essential files for reproducibility and submission

## 🔧 Main Components

### waste_classification_main.py

- Trains Custom CNN and MobileNetV2
- Evaluates both models
- Exports metrics to JSON
- Generates PNG visualizations

### generate_report_clean.py

- Generates JUTIF-compliant journal article
- All English technical terms italicized
- Tab indentation at paragraph starts
- Embeds visualizations automatically
- Creates professional academic layout

### waste_classification_cnn.ipynb

- Interactive notebook for exploration
- Step-by-step model development
- Visualization of results

## ✓ Verification

- ✓ Dataset: 8,400 images, 5 classes (balanced distribution)
- ✓ Models: Both trained and saved successfully
- ✓ Metrics: Complete evaluation results
- ✓ Report: Publication-ready format
- ✓ Clean: No redundant or duplicate files

## 📌 Notes

- All paths are configured relative to project root
- Models require ~14 GB disk space
- Training GPU recommended (CPU works but slower)
- Report generation is fast (< 1 minute)

---

**Last Updated:** January 4, 2026
**Status:** ✅ Clean & Ready for Submission
