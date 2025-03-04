# Assessment of Various Segmentation Techniques on EBHI-SEG Dataset

## Overview

This project explores various deep learning-based segmentation techniques applied to the **EBHI-SEG Dataset**. The primary goal is to compare the performance of different architectures in segmenting medical images and identifying relevant regions.

## Dataset

The **EBHI-SEG Dataset** consists of images categorized into different medical conditions, including:

- Adenocarcinoma
- High-grade IN
- Low-grade IN
- Normal
- Polyp
- Serrated adenoma

Each category contains corresponding image-label pairs for segmentation tasks.

## Models Used

The following pre-trained deep learning models are evaluated for segmentation:

- **U-Net**
- **ResNet50 U-Net**
- **MobileNet U-Net**
- **EfficientNet U-Net**
- **DenseNet U-Net**

## Methodology

1. **Data Preprocessing**

   - Images are resized to **224x224 pixels**.
   - Masks are converted to binary format.
   - Normalization is applied to images.

2. **Dataset Loading**

   - Custom function loads images and masks from structured directories.

3. **Training Setup**

   - Uses `train_test_split` for dataset division.
   - Implements **EarlyStopping** and **ReduceLROnPlateau** for optimized training.

4. **Evaluation Metrics**

   - **Dice Coefficient**
   - **Intersection over Union (IoU)**
   - **Mean Absolute Error (MAE)**
   - **Hausdorff Distance**
   - **Average Symmetric Surface Distance (ASSD)**
   - **DB Index**

## Installation & Usage

### Prerequisites

Ensure you have the following dependencies installed:

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

### Running the Notebook

To run the project, execute the Jupyter Notebook in a Kaggle or local Python environment:

```bash
jupyter notebook "Assessment of various segmentation techniques on EBHI SEG DATASET.ipynb"
```

## Results

Below is the performance comparison of different segmentation models:

| Model              | Accuracy | Precision | Recall | IoU    | Dice Coefficient | Mean IoU |
| ------------------ | -------- | --------- | ------ | ------ | ---------------- | -------- |
| U-Net              | 0.8010   | 0.8311    | 0.8627 | 0.7340 | 0.8466           | 0.6463   |
| ResNet50 U-Net     | 0.8172   | 0.8038    | 0.9430 | 0.7666 | 0.8679           | 0.6547   |
| MobileNet U-Net    | 0.6360   | 0.6388    | 0.9857 | 0.6329 | 0.7752           | 0.3280   |
| EfficientNet U-Net | 0.7542   | 0.7834    | 0.8485 | 0.6872 | 0.8146           | 0.5764   |
| DenseNet U-Net     | 0.8216   | 0.8160    | 0.9293 | 0.7683 | 0.8689           | 0.6658   |

## Contribution

Feel free to contribute by improving the model performance, optimizing hyperparameters, or adding new segmentation techniques.

