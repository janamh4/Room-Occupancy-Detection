Robust Room Occupancy Detection Using Deep Learning Under Sensor Failures

Project Overview

This project focuses on room occupancy detection using machine learning and deep learning techniques. It investigates model performance under normal operating conditions and evaluates robustness when environmental sensor data is affected by simulated sensor failures.

The project uses the UCI Occupancy Detection Dataset to classify room occupancy into two classes:

 * 0: Unoccupied
 * 1: Occupied

The main objective is to compare different models and investigate whether sensor masking during training can improve robustness against sensor failures.

Dataset

The dataset contains 20,560 records collected from an office environment.

It includes the following environmental features:

* Temperature
* Relative Humidity
* Light
* CO2
* Humidity Ratio

The target variable is Occupancy.

The dataset is divided into:

* Training set: 8,143 original records, split chronologically into 80% training and 20% validation.
* Test 1: 2,665 records.
* Test 2: 9,752 records.

The training data is sorted chronologically before splitting.

Data Preprocessing

The following preprocessing steps were performed:

* Checked missing values and duplicate records.
* Selected five environmental features.
* Separated the input features from the target variable.
* Applied chronological 80/20 training-validation splitting.
* Applied StandardScaler normalization.
* Fitted the scaler using training data only and applied it to validation and test sets.
* Saved prepared data, raw data, scaler, and dataset metadata to support reproducibility.

Models Implemented

Three model configurations were implemented and evaluated.

1. Standard Multi-Layer Perceptron (MLP)

The MLP architecture consists of:

* Input layer: 5 features.
* First hidden layer: 64 neurons with ReLU activation.
* Second hidden layer: 32 neurons with ReLU activation.
* Output layer: 1 neuron with Sigmoid activation.

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

3. Sensor-Masked MLP

A second MLP model was trained using sensor-masked data to investigate robustness against sensor failures.

The model uses the same architecture and training configuration as the standard MLP.

Sensor Masking Strategy:

* Original training samples were combined with additional masked samples.
* One sensor was randomly selected for each additional sample.
* The selected sensor’s corresponding feature values were replaced with zero.
* The augmented dataset was used to train the model.

This approach helps the model learn from simulated sensor failure conditions.

Sensor Failure Experiments

Four sensor failure scenarios were evaluated:

* Temperature Failure
* Humidity Failure
* Light Failure
* CO2 Failure

Each failure was simulated by replacing the corresponding standardized feature values with zero.

The Sensor-Masked MLP was evaluated under these conditions on both Test 1 and Test 2.

Evaluation Metrics

Models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

Evaluation was performed separately on Test 1 and Test 2.

Experimental Results

Performance with All Sensors Available

| Model	            | Test Set	| Accuracy	| Precision	| Recall	| F1-score|
|---|---|---:|---:|---:|---:|
| Standard MLP	     | Test 1	  | 97.07%  	| 92.65%	   | 99.90%	| 96.14%  |
| Standard MLP	     | Test 2	  | 92.07%  	| 74.20%	   | 95.46%	| 83.50%  |
| Sensor-Masked MLP	| Test 1	  | 96.44%	  | 91.17%	   | 99.90%	| 95.34%  |
| Sensor-Masked MLP	| Test 2	  | 88.69%	  | 65.04%	   | 99.80%	| 78.76%  |
| Random Forest	    | Test 1	  | 92.61%	  | 94.08%	   | 85.08%	| 89.36%  |
| Random Forest    	| Test 2	  | 96.87%	  | 94.49%	   | 90.39%	| 92.39%  |

Sensor Failure Results

The following table presents the F1-scores of the Sensor-Masked MLP under individual sensor failures.

| Test Set	| Failed Sensor	| Accuracy	| Precision	| Recall	| F1-score |
|---|---|---:|---:|---:|---:|
| Test 1	  | Temperature	  | 97.07%	  | 92.65%	   | 99.90%	| 96.14%   |
| Test 1	  | Humidity	     | 94.37%	  | 86.70%	   | 99.90%	| 92.83%   |
| Test 1	  | Light	        | 86.08%	  | 74.33%	   | 94.44%	| 83.19%   |
| Test 1	  | CO2	          | 97.94%	  | 94.64%	   | 100.00%	| 97.25%  |
| Test 2	  | Temperature	  | 86.68%	  | 61.22%	   | 99.85%	| 75.90%   |
| Test 2	  | Humidity	     | 84.78%	  | 58.01%	   | 99.85%	| 73.39%   |
| Test 2	  | Light	        | 72.25%	  | 38.82%	   | 55.69%	| 45.75%   |
| Test 2	  | CO2	          | 98.17%	  | 92.27%	   | 99.66%	| 95.82%   |

Results Discussion

The experimental results show that model performance varies across different test sets and sensor conditions.

* The Standard MLP achieved the highest F1-score on Test 1 when all sensors were available.
* Random Forest achieved the best overall performance on Test 2 with all sensors available.
* The Sensor-Masked MLP showed improvements under certain sensor failure conditions, particularly Light and CO2 failures.
* On Test 2, CO2 failure resulted in an F1-score of 95.82% for the Sensor-Masked MLP.
* However, sensor masking did not improve performance consistently across all sensors and test sets.

These findings highlight the importance of evaluating model robustness under different sensor failure scenarios rather than relying only on normal operating conditions.

Repository Contents
*
 Room_Occupancy_Final_Reproducibility.ipynb: Notebook containing data preprocessing, model implementation, training, evaluation, sensor failure experiments, and reproducibility checks.
* README.md: Project documentation.
* requirements.txt: Required Python libraries.

Technologies Used

* Python
* TensorFlow / Keras
* Scikit-learn
* NumPy
* Pandas
* Matplotlib
* Joblib

Dataset Source

UCI Machine Learning Repository – Occupancy Detection Dataset.

https://archive.ics.uci.edu/dataset/357/occupancy+detection
