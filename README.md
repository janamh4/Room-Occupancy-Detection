Robust Room Occupancy Detection Using Deep Learning Under Sensor Failures

Project Overview

This project focuses on room occupancy detection using machine learning and deep learning techniques. It aims to investigate model performance and robustness when sensor data is affected by failures.

The project uses the UCI Occupancy Detection Dataset to classify room occupancy into two classes:

* 0: Unoccupied
* 1: Occupied

Dataset

The dataset contains readings from the following environmental sensors and features:

* Temperature
* Relative Humidity
* Light
* CO2
* Humidity Ratio

The target variable is Occupancy.

Models Implemented

Two models were implemented and evaluated:

1. Standard Multi-Layer Perceptron (MLP)

The MLP architecture consists of:

* Input layer: 5 features
* First hidden layer: 64 neurons with ReLU activation
* Second hidden layer: 32 neurons with ReLU activation
* Output layer: 1 neuron with Sigmoid activation

Training Configuration:

* Optimizer: Adam
* Learning Rate: 0.001
* Loss Function: Binary Crossentropy
* Batch Size: 32
* Maximum Epochs: 50
* Actual Epochs: 8
* Early Stopping: Patience = 7
* Class Weights: Balanced
* Classification Threshold: 0.3

2. Random Forest

Random Forest was implemented as a baseline model for comparison.

Configuration:

* Number of Estimators: 100
* Random State: 42
* Class Weights: Balanced
* Number of Jobs: -1

Evaluation Metrics

Both models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score

Evaluation was performed separately on Test 1 and Test 2.

Experimental Results

Model	         | Test Set	|  Accuracy |	 Precision	|  Recall	|  F1-score
__________________________________________________________________________
MLP	          |   Test 1	|   97.07%  |   92.65%	  |  99.90%	|   96.14%
Random Forest |   Test 1  |   92.61%	|   94.08%	  |  85.08%	|   89.36%
MLP	          |   Test 2	|   91.28%	|   72.53%    |  94.19% |   81.95%
Random Forest	|   Test 2	|   96.87%	|   94.49%	  |  90.39% |   92.39%

Results Discussion

The MLP achieved better Accuracy, Recall, and F1-score on Test 1, while Random Forest achieved higher Precision.

On Test 2, Random Forest achieved better Accuracy, Precision, and F1-score. However, MLP achieved higher Recall.

These results demonstrate that model performance varies across test sets, highlighting the importance of evaluating models under different data conditions.

Repository Contents

* Room_Occupancy.ipynb: Notebook containing the implementation, training, and evaluation of the MLP and Random Forest models.
* README.md: Project documentation.
* requirements.txt: Required Python libraries.

Technologies Used

* Python
* TensorFlow / Keras
* Scikit-learn
* NumPy
* Pandas
* Matplotlib

Dataset Source

UCI Machine Learning Repository – Occupancy Detection Dataset.

https://archive.ics.uci.edu/dataset/357/occupancy+detection
