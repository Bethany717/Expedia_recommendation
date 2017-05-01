import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time


start = time.time()
inf = open("../data/data_cleaned.csv")
data = np.array([map(float, s.strip().split(',')) for s in inf.readlines()])

X = data[:, 0 : -1]
Y = data[:, -1 ]

print X.shape
print Y.shape


seed = 7
use_size = 0

X_use, X_not, Y_use, Y_not = train_test_split(X, Y, test_size=use_size, random_state=seed)

print X_use.shape
print X_not.shape

test_size = 0.30

X_train, X_test, y_train, y_test = train_test_split(X_use, Y_use, test_size=test_size, random_state=seed)
print X_train.shape
print y_train.shape
print X_test.shape
model = xgb.XGBClassifier()
model.fit(X_train, y_train)
print model

print time.time()- start
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
#evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

print time.time() - start
