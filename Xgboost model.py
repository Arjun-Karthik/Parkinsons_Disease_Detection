#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

#Load Dataset
dataset = pd.read_csv('parkinsons.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Handling Data
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = ct.fit_transform(X)
print(X)

# Ensure X is a sparse matrix
X_sparse = csr_matrix(X)

#Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X_sparse, y, test_size=0.2, random_state=0)

#Load Model
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

#Prediction
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:",cm)
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
report = classification_report(y_test, y_pred)
print(report)