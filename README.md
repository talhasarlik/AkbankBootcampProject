# **Akbank Global AI Hub Bootcamp Project**
# Fish Species Classification Using ANN

This project uses an Artificial Neural Network (ANN) to classify fish species from a large-scale fish dataset. The dataset consists of images of fish, and the model is trained to categorize these images into various fish species.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)

## Project Overview

This project applies an Artificial Neural Network (ANN) for image classification. The dataset used consists of fish images categorized into multiple species. The model preprocesses the input images, trains on a labeled dataset, and is evaluated on unseen data.

### Key Features:
- ANN-based model for image classification.
- Preprocessing and data augmentation using `ImageDataGenerator`.
- Evaluation through accuracy, loss, confusion matrix, and classification report.

## Dataset

The dataset is sourced from **Kaggle**: [A Large Scale Fish Dataset](https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset).

It contains several classes of fish images, which are used to train, validate, and test the ANN model.

### Dataset Structure:
- The dataset images are in `.png` format.
- Each class represents a different species of fish.

## Model Architecture

The model is a basic ANN (Artificial Neural Network) consisting of the following layers:
- **Input Layer:** 224x224 image input flattened to a 1D vector.
- **Dense Layers:** Two hidden layers with ReLU activation.
- **Dropout Layers:** To prevent overfitting.
- **Output Layer:** Softmax activation to classify fish into one of 9 species.

### Optimizer:
- **Adam** optimizer is used with a learning rate scheduler to adjust learning rate during training.

### Loss Function:
- `categorical_crossentropy` is used as the loss function.

## Installation

### Prerequisites
- Python 3.x
- TensorFlow
- Scikit-learn
- Matplotlib
- Seaborn
- Pandas
- NumPy

### Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/talhasarlik/AkbankBootcampProject.git
   cd AkbankBootcampProject
   ```

2. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset from Kaggle:
   - The dataset can be downloaded from Kaggle using the `kagglehub` API.

4. Run the notebook:
   The project is implemented in a Jupyter Notebook format. Open the notebook and execute the cells in order.

## Usage

1. Ensure that the dataset is correctly downloaded and stored in the right directory.
2. Run the code to preprocess the data and train the ANN model.

### Training the Model:

To train the model, you can run the following code in your Jupyter Notebook:

```python
history = model.fit(
    train_images, 
    validation_data=val_images, 
    epochs=20, 
    callbacks=[lr_schedule, early_stopping]
)
```

### Evaluation:

The trained model can be evaluated using test data:

```python
test_loss_acc = model.evaluate(test_images)
print('Test loss is:', test_loss_acc[0])
print('Test accuracy is:', test_loss_acc[1]*100, '%')
```

### Results Visualization:

You can visualize the training history and confusion matrix:

```python
# Accuracy graph for training and validation
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.legend()
plt.title('Accuracy: Training vs Validation')
plt.show()

# Confusion Matrix
sns.heatmap(confusion_matrix(predict_data['label'], predict_data['pred']), annot=True, fmt='2d')
```

## Results

The model is trained on the fish dataset, and its performance is evaluated based on accuracy, loss, confusion matrix, and classification report. The following metrics are captured:

- **Training Accuracy:** 90.22%
- **Validation Accuracy:** 90.14%
- **Test Accuracy:** 90.72%
