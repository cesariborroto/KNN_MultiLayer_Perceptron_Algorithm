# K-Nearest Neighbors (KNN) Algorithm with Wine Dataset

This repository contains a Jupyter Notebook file implementing the K-Nearest Neighbors (KNN) algorithm for classification. The algorithm is applied to the Wine dataset, which is a popular dataset used in machine learning.

## Dataset
The Wine dataset is used in this notebook, which can be downloaded from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine). The dataset consists of various chemical analysis measurements of three different cultivators of wine. The goal is to classify the cultivator based on the chemical features.

## Prerequisites
- Python 3.x
- Jupyter Notebook
- Required libraries: `pandas`, `numpy`, `matplotlib`, `scikit-learn`

## Algorithm Overview
The KNN algorithm is a simple and effective classification algorithm. It classifies new data points based on their similarity to known data points in the training set. The algorithm calculates the distance between the new data point and all other data points in the training set. It then selects the K nearest neighbors and assigns the class label based on majority voting.

## Notebook Contents
1. Importing Required Libraries: The necessary libraries, such as `pandas`, `numpy`, `matplotlib`, and `scikit-learn`, are imported.
2. Loading the Dataset: The Wine dataset is loaded from a CSV file using `pandas`. The top 5 rows and statistical values of the dataset are displayed.
3. Data Preprocessing: The dataset is divided into features (`X`) and the target column (`y`). The data is further split into training and testing sets using `train_test_split` from `scikit-learn`.
4. Data Scaling: The data is scaled using `StandardScaler` from `scikit-learn` to ensure all features have the same scale.
5. KNN Algorithm Implementation: The KNN algorithm is implemented using `KNeighborsClassifier` from `scikit-learn`. The algorithm is trained on the training set and tested on the testing set.
6. Model Evaluation: The performance of the KNN algorithm is evaluated by generating a confusion matrix and classification report using `classification_report` and `confusion_matrix` from `scikit-learn`.

Feel free to explore and modify the notebook according to your needs.
