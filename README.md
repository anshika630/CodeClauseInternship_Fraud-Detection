%pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn

import pandas as pds
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler as scalar
from sklearn.ensemble import RandomForestClassifier as LR
from sklearn.metrics import classification_report as cr, confusion_matrix as cm, roc_auc_score as ras
from imblearn.over_sampling import SMOTE

data = pds.read_csv("creditcard.csv")
data.head()

print("Dataset shape:", data.shape)
print("Missing values:\n", data.isnull().sum())
print("Class distribution:\n", data['Class'].value_counts())

sc = scalar()
data['Amount'] = sc.fit_transform(data[['Amount']])
data = data.drop(['Time'], axis=1)

print(y_tr.value_counts(normalize=True))
print(y_ts.value_counts(normalize=True))
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_ts_scaled = scaler.transform(X_ts)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_tr_scaled, y_tr)
y_pred = model.predict(X_ts_scaled)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_ts, y_pred))
print(classification_report(y_ts, y_pred))
