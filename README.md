# CodeClauseInternship_Fraud-Detection

A. Project Overview
This project demonstrates:

a.How to handle imbalanced datasets using SMOTE
b.Basic data preprocessing and scaling
c. Training a Random Forest Classifier for binary classification
d. Exploratory analysis of class distribution and dataset structure

It is a clean starting point for learning fraud detection pipelines in Python + scikit-learn.

B. Project Structure
fraud-detection-project/
│
├── Fraud Detection.ipynb      # Main Jupyter Notebook
├── creditcard.csv             # Dataset (must be added manually)
├── requirements.txt           # Dependencies for easy setup
├── README.md                  # Project documentation

C.Dataset
Kaggle Credit Card Fraud Detection dataset (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud):
1.284,807 transactions
2.492 fraud cases (~0.17%)
3.PCA-transformed anonymized features V1–V28
4.Amount: Transaction amount
5.Time: Time since first transaction (dropped)
6.Class: Target (1 = fraud, 0 = normal)

D.Getting Started

1. Clone the repository
  git clone https://github.com/anshika630/fraud-detection-project.git
  cd fraud-detection-project

2. Install dependencies
   pip install -r requirements.txt

3. Download the dataset- https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

4. Run the Notebook
   jupyter notebook "Fraud Detection.ipynb"

E. Workflow

1. Install required libraries (numpy, pandas, matplotlib, seaborn, scikit-learn, imbalanced-learn)
   
2. Load dataset and check:
a.Shape
b.Missing values
c.Class distribution

3.Preprocessing:
a.Scale Amount using StandardScaler
b.Drop Time column

4.Handle Class Imbalance:
a.Apply SMOTE to oversample minority class

5.Train Random Forest Classifier

6.Inspect feature correlations (optional)

7.Print data distribution and processed shapes

Learning Outcomes:
Loading and exploring a real-world dataset
1. Applying SMOTE for balancing datasets
2. Preprocessing using scaling and dropping unneeded features
3. Basic Random Forest model training on imbalanced data
4. Understanding the workflow of fraud detection modeling pipelines
