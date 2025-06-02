# UCI Heart Disease Neural Network Classification 

This project implements a shallow neural network using PyTorch to predict the likelihood of heart disease in patients, based on the "Heart Disease UCI" dataset. The performance of the neural network is then compared against several traditional machine learning models.

## Project Objectives

The main goals of this project are:
1.  To implement and train a shallow neural network (one to three hidden layers) using PyTorch.
2.  To evaluate the neural network's performance in predicting heart disease using the UCI dataset.
3.  To compare the neural network's performance against other established machine learning models.
4.  To investigate the impact of architectural choices (number of hidden layers, neurons per layer, learning rate, batch size) on the neural network's performance.

---

## Dataset Overview

* **Source:** Heart Disease UCI dataset
* **Instances:** 920
* **Features:** 14 relevant attributes (after dropping 'id' and 'dataset' implicitly before preprocessing for X). Key features include:
    * `age`: Age in years
    * `sex`: Sex (e.g., Male/Female)
    * `cp`: Chest pain type
    * `trestbps`: Resting blood pressure (mm Hg)
    * `chol`: Serum cholesterol (mg/dl)
    * `fbs`: Fasting blood sugar > 120 mg/dl (Boolean)
    * `restecg`: Resting electrocardiographic results
    * `thalch`: Maximum heart rate achieved
    * `exang`: Exercise induced angina (Boolean)
    * `oldpeak`: ST depression induced by exercise relative to rest
    * `slope`: The slope of the peak exercise ST segment
    * `ca`: Number of major vessels (0-3) colored by fluoroscopy
    * `thal`: Thallium stress test result
* **Target Variable:** `num` (representing the presence/severity of heart disease, treated as a multi-class classification problem with 5 classes: 0, 1, 2, 3, 4).

---

## Project Workflow

1.  **Setup & Environment:**
    * Imported necessary libraries including PyTorch, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, and Imbalanced-learn.
    * Configured device for PyTorch (CUDA if available).

2.  **Data Loading & Initial Exploration:**
    * Loaded the dataset from `heart_disease_uci.csv`.
    * Inspected data shape, info, descriptive statistics, and head.
    * Dropped the 'id' column.
    * Visualized class distribution of the target variable (`num`).
    * Examined feature correlations using a heatmap.

3.  **Data Preprocessing:**
    * Identified categorical and numerical columns.
    * **Missing Value Imputation:**
        * Numerical features: `IterativeImputer`.
        * Categorical features: `SimpleImputer` (strategy='most_frequent').
    * **Encoding Categorical Features:** `OneHotEncoder` was applied to categorical features.
    * **Scaling Numerical Features:** `StandardScaler` was applied to numerical features.
    * Concatenated preprocessed numerical and encoded categorical features.
    * **Handling Class Imbalance:** `SMOTE` (Synthetic Minority Over-sampling Technique) was used to resample the training data and address class imbalance.
    * **Train-Test Split:** The resampled data was split into training (80%) and testing (20%) sets, stratified by the target variable.

4.  **Neural Network (PyTorch):**
    * **Dataset & DataLoader:** Custom PyTorch `Dataset` and `DataLoader` were created for training and testing.
    * **Model Definition (`HeartDiseaseNet`):** A feedforward neural network with:
        * An input layer (size based on features).
        * Three hidden layers with ReLU activation functions (sizes experimented: (128, 64, 32), (256, 128, 64), (64, 32, 16)).
        * An output layer with 5 neurons (for 5 classes).
    * **Loss Function:** `CrossEntropyLoss` with class weights to further address imbalance.
    * **Optimizer:** `Adam`.
    * **Training:** The model was trained for a set number of epochs (100 for initial training, 30 for experiments).
    * **Hyperparameter Experiments:** Systematically varied hidden layer sizes, learning rates (1e-3, 5e-4, 1e-4), and batch sizes (16, 32, 64) to observe their impact on Macro F1-score. The configuration (256, 128, 64) hidden layers, LR=0.001, Batch Size=32 yielded the best Macro F1 (0.8751) in experiments.

5.  **Model Evaluation:**
    * **Neural Network:** Evaluated on the test set using accuracy, classification report (precision, recall, F1-score per class), and confusion matrix. The initial model achieved a test accuracy of approximately 86.62%.
    * **Baseline Traditional Models:** Several Scikit-learn classifiers were trained on the SMOTE-balanced data and evaluated:
        * Logistic Regression
        * Linear Discriminant Analysis (LDA)
        * Decision Tree Classifier
        * Random Forest Classifier
        * Ensemble (Voting Classifier with LR, RF, SVC)
        * K-Nearest Neighbors (KNN)
        * Gaussian Naive Bayes
        * Support Vector Machine (SVM)
    * Metrics: Accuracy, Macro F1-score, Classification Report.

---

## Key Results & Discussion

* The implemented **PyTorch neural network** demonstrated the highest performance among the models tested, achieving a test accuracy of **~86.62%** and a top Macro F1 score of **0.8751** during hyperparameter experiments.
* Among the traditional machine learning models, **Random Forest** and the **Ensemble Voting Classifier** showed strong comparative performance.
* Models like Logistic Regression, LDA, and Naive Bayes underperformed, likely due to the complex, non-linear nature of the dataset and strong independence assumptions (for Naive Bayes).
* Hyperparameter experiments revealed that network architecture (hidden layer sizes), learning rate, and batch size significantly impact the neural network's performance.

---

## Technologies & Libraries Used

* **Python 3**
* **PyTorch:** For neural network implementation.
* **Scikit-learn:** For data splitting, preprocessing, traditional ML models, metrics, and class weight computation.
* **Imbalanced-learn (imblearn):** For SMOTE.
* **Pandas:** For data manipulation.
* **NumPy:** For numerical operations.
* **Matplotlib & Seaborn:** For data visualization.

