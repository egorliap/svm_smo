import numpy as np
import pandas as pd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    confusion_matrix,
)
from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC

from scr.svm import SVMClassifier

file_path = "datasets/diabetes_dataset.csv"

df = pd.read_csv(file_path)

X = df.drop(
    [
        "Outcome",
    ],
    axis=1,
)
y = df["Outcome"]
y = y.replace(0, -1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=10
)

scaler = StandardScaler()

scaled_X_train = np.array(scaler.fit_transform(X_train))
scaled_X_test = np.array(scaler.transform(X_test))


svc_model = SVMClassifier(C=1, max_iter=1000)
svc_model.fit(scaled_X_train, np.array(y_train))

svc_pred = svc_model.predict(np.array(scaled_X_test))

y_test = np.array(y_test)
w = svc_model.support()

svc_model.support()

print(classification_report(y_test, svc_pred))


svc_model = SVC(kernel="linear")
svc_model.fit(scaled_X_train, y_train)

svc_pred = svc_model.predict(scaled_X_test)

print(classification_report(y_test, svc_pred))
