# Waste Classification CNN

A Convolutional Neural Network (CNN) model for automatic waste classification into 5 categories: food waste, glass, metal, paper, and plastic.

## Project Overview

This project implements a deep learning solution using CNN to classify different types of waste from images. The model is designed for automatic waste sorting systems and has been trained on a dataset of 8,400 training images across 5 waste categories.

## Dataset

The dataset is organized into three sets:

- **Training**: 8,400 images
- **Validation**: 469 images
- **Test**: 237 images

### Classes

- Food Waste
- Glass
- Metal
- Paper
- Plastic

Dataset location: `datasets/`

## Model Architecture

The CNN model consists of:

- 3 Convolutional layers with filters: 16, 32, 32
- MaxPooling2D after each convolutional layer
- Flatten layer
- Dense layer with 32 units + Dropout (0.3)
- Output layer with 5 units (softmax activation)
- Total parameters: 80,069

## Model Performance

- **Test Accuracy**: 61.60%
- **Test Precision**: 58.98%
- **Test Recall**: 61.60%
- **Test F1-Score**: 58.07%

## Training Configuration

- **Optimizer**: Adam (learning_rate=0.001)
- **Loss Function**: Sparse Categorical Crossentropy
- **Batch Size**: 16
- **Epochs**: 8
- **Callbacks**: EarlyStopping, ReduceLROnPlateau
- **Image Size**: 64x64 pixels
- **Normalization**: Pixel values scaled to 0-1

## Files Structure

```
CV_Unpam/
├── README.md
├── waste_classification_cnn.ipynb
├── datasets/
│   ├── train/
│   ├── valid/
│   └── test/
├── models/
│   └── waste_classification_model.h5
└── report/
    └── waste_classification_report.html
```

## Usage

The complete implementation and training code is available in `waste_classification_cnn.ipynb`.

### Requirements

- Python 3.x
- TensorFlow/Keras
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

### Running the Notebook

1. Ensure all dependencies are installed
2. Open `waste_classification_cnn.ipynb` in Jupyter Notebook
3. Execute the cells in order to train and evaluate the model

## Model Improvements

Future enhancements to improve model performance:

1. **Data Augmentation**: Apply rotation, flip, and zoom transformations to increase data variety
2. **Regularization**: Add more dropout layers or batch normalization to reduce overfitting
3. **Architecture Optimization**: Experiment with deeper networks or pre-trained models
4. **Hyperparameter Tuning**: Optimize learning rate, batch size, and other parameters

## Reports

Detailed analysis and results are available in `report/waste_classification_report.html`.

## License

This project is part of the Computer Vision course at UNPAM.
