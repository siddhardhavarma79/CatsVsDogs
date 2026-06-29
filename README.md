# Cat and Dog Image Classification using Support Vector Machine (SVM)

## Overview

This project implements a Support Vector Machine (SVM) model to classify images of cats and dogs. The images are preprocessed using OpenCV, converted to grayscale, resized, and flattened into feature vectors before training the SVM classifier.

## Objective

The objective of this project is to build an image classification system capable of distinguishing between cats and dogs using machine learning techniques.

## Dataset

Dataset: Cat and Dog Dataset from Kaggle

The dataset contains images of cats and dogs organized into separate folders for training and testing.

## Technologies Used

* Python
* NumPy
* OpenCV
* Scikit-learn

## Project Workflow

1. Load cat and dog images from the dataset.
2. Resize images to 32×32 pixels.
3. Convert images to grayscale.
4. Flatten images into feature vectors.
5. Split the dataset into training and testing sets.
6. Train a Support Vector Machine (SVM) classifier.
7. Evaluate model performance using accuracy score and classification report.

## Project Structure

```text
cat-and-dog/
│
├── svm_cat_dog.py
├── requirements.txt
├── README.md
├── training_set/
└── test_set/
```

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Run the Project

```bash
python svm_cat_dog.py
```

## Expected Output

* Dataset loading information
* Model training progress
* Accuracy score
* Classification report
* Sample image prediction

## Results

The SVM classifier successfully classifies cat and dog images based on extracted image features. The model performance is evaluated using accuracy and classification metrics.

## Skills Demonstrated

* Image Processing
* Machine Learning
* Support Vector Machines (SVM)
* Data Preprocessing
* Model Evaluation
* Python Programming

## Author

VALISHETTI SIDDHARDHA

B.Tech – Artificial Intelligence and Machine Learning

SkillCraft Technology AI/ML Internship
