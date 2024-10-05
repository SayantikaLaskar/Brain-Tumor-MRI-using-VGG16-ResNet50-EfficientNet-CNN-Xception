# Brain-Tumor-MRI-using-VGG16-ResNet50-EfficientNet-CNN-Xception


Welcome to the **Brain Tumor MRI Classification** project! This repository demonstrates the use of advanced Convolutional Neural Networks (CNNs) to classify brain MRI images into tumor and non-tumor categories. The models utilized include state-of-the-art architectures like **EfficientNet**, **ResNet50**, **Xception**, and **VGG16** to achieve high accuracy and performance in image classification.

---

## 🎯 Project Overview

The goal of this project is to classify brain MRI scans into **tumor** or **non-tumor** categories using pre-trained CNN models. The dataset used contains MRI images of brain tumors, and the models are trained and evaluated to help improve automated diagnostics in medical imaging 🏥.

### 🔍 Key Highlights:
- **Multiple CNN Models**: We explore and fine-tune several popular CNN architectures including:
  - EfficientNet
  - ResNet50
  - Xception
  - VGG16
- **Data Augmentation**: We implement powerful data augmentation strategies to increase the variability of training data and prevent overfitting.
- **Evaluation Metrics**: The models are evaluated on a wide range of metrics, including:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - Confusion Matrix 📊
  - AUC-ROC Curve 📈
  - Support (class distribution)
- **Visualizations**: Comprehensive visualizations for confusion matrices, loss/accuracy plots, and sample predictions.

---

## 📂 Dataset

The **Brain Tumor MRI Dataset** used in this project is sourced from [Kaggle](https://www.kaggle.com/masoudnickparvar/brain-tumor-mri-dataset). It consists of:
- **Training Data**: Brain MRI images categorized into `tumor` and `non-tumor`.
- **Testing Data**: Separate set of MRI images used for testing model performance.

---

## 🛠️ Technologies and Tools

The following technologies and libraries are used in this project:
- **Deep Learning Framework**: TensorFlow, Keras
- **Data Handling**: Pandas, Numpy
- **Image Processing**: PIL, OpenCV
- **Visualization**: Matplotlib, Plotly
- **Evaluation**: Scikit-learn

### 📦 Python Packages
To install the necessary dependencies, run:
```bash
pip install -r requirements.txt
```

### Requirements
- `tensorflow`
- `keras`
- `scikit-learn`
- `matplotlib`
- `plotly`
- `pandas`
- `numpy`

---

## 🚀 Models and Training

### 1. **EfficientNet**
EfficientNet is a family of neural networks that are optimized for performance and efficiency. In this project, EfficientNet is fine-tuned to classify MRI images into tumor/non-tumor categories.

### 2. **ResNet50**
ResNet50 is a deep residual network that overcomes the vanishing gradient problem by using skip connections. It allows the model to go deeper with better accuracy.

### 3. **Xception**
Xception is a powerful model built on the depthwise separable convolution idea. It's highly efficient for image classification tasks and is fine-tuned for our specific dataset.

### 4. **VGG16**
VGG16 is a classic CNN architecture that uses a stack of convolutional layers followed by fully connected layers for classification.

Each model is trained using the **Adamax optimizer** and evaluated on the test set. The models are compared based on accuracy, precision, recall, and F1 scores.

---

## 📊 Evaluation Metrics

The models are evaluated using the following metrics:
1. **Accuracy**: Overall accuracy of the model in classifying the images.
2. **Precision**: How many of the predicted positive cases were correct.
3. **Recall**: How many of the actual positive cases were correctly identified.
4. **F1 Score**: A balance between precision and recall.
5. **Confusion Matrix**: To visualize the model's performance in a matrix form.
6. **AUC-ROC Curve**: Evaluates the true positive rate against the false positive rate across thresholds.




## 📜 Results

The table below summarizes the performance of each model across different evaluation metrics:

| Model       | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|-------------|----------|-----------|--------|----------|---------|
| EfficientNet| 96.3%    | 0.96      | 0.95   | 0.98     | 0.97    |
| ResNet50    | 94.8%    | 0.94      | 0.93   | 0.93     | 0.96    |
| Xception    | 89.5%    | 0.95      | 0.94   | 0.94     | 0.96    |
| VGG16       | 97.3%    | 0.92      | 0.91   | 0.91     | 0.94    |

---

## 👏 Acknowledgments

Special thanks to the [Kaggle community](https://www.kaggle.com/) for providing the dataset and to the developers of TensorFlow and Keras for their powerful tools for deep learning.
