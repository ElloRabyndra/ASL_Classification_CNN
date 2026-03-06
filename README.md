# Sign Language Letters and Numbers Classification Using CNN (MobileNetV2 Transfer Learning)

## Overview

This project implements a deep learning-based classification system for American Sign Language (ASL) hand gestures, covering both **letters (A-Z)** and **digits (0-9)**. The model is built using Transfer Learning with MobileNetV2 architecture, trained on a combined dataset of 34 classes (24 letters + 10 digits).

This project was developed as a final assignment for a Computer Vision course.

## Dataset

Two public datasets from Kaggle are combined:

| Dataset                      | Classes                       | Image Size      | Format       | Source                                |
| ---------------------------- | ----------------------------- | --------------- | ------------ | ------------------------------------- |
| Sign Language MNIST          | 24 letters (A-Z, excl. J & Z) | 28x28 grayscale | CSV          | datamunge/sign-language-mnist         |
| Sign Language Digits Dataset | 10 digits (0-9)               | 64x64 grayscale | NumPy (.npy) | ardamavi/sign-language-digits-dataset |

> Note: Letters J and Z are excluded because they require motion gestures and cannot be represented as static images.

Place the dataset files in the following structure before running the notebook:

```
comvis-sign-language/
├── sign_mnist_train.csv
├── sign_mnist_test.csv
├── X.npy
└── Y.npy
```

## Pipeline

```
Load Dataset 1 (Sign MNIST)  +  Load Dataset 2 (Digits .npy)
                    |
           Resize & Normalization
                    |
       Exploratory Data Analysis (EDA)
                    |
    Split Train / Validation / Test (70/15/15)
                    |
         Training Data Augmentation
                    |
   Transfer Learning: MobileNetV2 + Custom Head
                    |
     Phase 1: Frozen Base  -->  Phase 2: Fine-Tuning
                    |
      Evaluation: Accuracy, F1, Confusion Matrix
                    |
       Misclassification Analysis
```

## Model Architecture

- **Base model**: MobileNetV2 pretrained on ImageNet (frozen during Phase 1)
- **Custom head**: GlobalAveragePooling2D -> Dense(256) -> Dropout(0.4) -> Dense(128) -> Dropout(0.3) -> Dense(34, softmax)
- **Training strategy**: Two-phase training (frozen base then fine-tuning last 30 layers)
- **Input size**: 64x64x3

## Requirements

```
tensorflow
scikit-learn
pandas
numpy
matplotlib
seaborn
opencv-python
```

Install all dependencies:

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn opencv-python
```

## How to Run

**Google Colab**

1. Upload the dataset to Google Drive under `MyDrive/comvis-sign-language/`
2. Open the notebook in Colab
3. Enable GPU: Runtime > Change runtime type > T4 GPU
4. Run all cells from top to bottom

**VSCode / Local**

1. Install dependencies using the command above
2. In the notebook, replace the `DRIVE_PATH` variable with your local dataset path:

```python
   DRIVE_PATH = 'your/local/path/to/comvis-sign-language/'
```

3. Remove the Google Drive mount cell
4. Run the notebook

## Results

The model is evaluated using the following metrics computed on the test set:

- Accuracy
- Precision, Recall, F1-Score (per class and macro average)
- Confusion Matrix

Detailed results and misclassification analysis are available inside the notebook.

## Project Structure

```
.
├── data/
│   ├── sign_mnist_train.csv
│   ├── sign_mnist_test.csv
│   ├── X.npy
│   └── Y.npy
├── ASL_Classification_CNN.ipynb
└── README.md
```

## References

- Howard, A. G., et al. (2017). MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications.
- Sandler, M., et al. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. CVPR.
- Sign Language MNIST: https://www.kaggle.com/datasets/datamunge/sign-language-mnist
- Sign Language Digits Dataset: https://www.kaggle.com/datasets/ardamavi/sign-language-digits-dataset
