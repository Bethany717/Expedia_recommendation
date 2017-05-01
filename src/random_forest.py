from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import ml_metrics as metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import operator
from sklearn.metrics import confusion_matrix
import seaborn as sn


def plot(data, verbose):
    if verbose:
        sorted_x = data
        sorted_x.sort(key=lambda tup: tup[1])
        print  sorted_x

        Xaxis = np.arange(35)
        data = [sorted_x[i][1] for i in range(len(sorted_x))]
        name = [sorted_x[i][0] for i in range(len(sorted_x))]

        plt.figure()
        axl = plt.axes([0.3, 0.1, 0.65, 0.82])


        axl.barh(Xaxis, data, align='center', alpha=0.8)
        axl.set_yticks(Xaxis)
        axl.set_yticklabels(name)
        axl.set_title("Feature importance")

        plt.show()

    else:
        pass

def plot_confusion_matrix(data):
    df_cm = pd.DataFrame(data, index=[i for i in range(10)],
                         columns=[i for i in range(10)])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)
    plt.title("Confusion Matrix Obtained from Random Forest Prediction")
    plt.show()

inf = open("../data/data_cleaned.csv")
feat_names = inf.readline().strip().split(',')
data = np.array([map(float, s.strip().split(',')) for s in inf.readlines()])

X = data[:, 0: -1]
Y = data[:, -1]

print X.shape
print Y.shape

seed = 7

test_size = 0.10

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
print X_train.shape
print y_train.shape
print X_test.shape

test = [120, 140, 160, 180, 200]
for i in test:
    print i
    clf = RandomForestClassifier(n_estimators=i, max_depth=25, min_samples_leaf=1)
    clf.fit(X_train, y_train)
    preds = clf.predict_proba(X_test)
    predicted = preds.argsort(axis=1)[:, -np.arange(1, 6)]

    importance = zip(feat_names[: -1], clf.feature_importances_)
    #print importance
    #plot(importance, True)

    print("score:", metrics.mapk(y_test.astype(np.int64).reshape(-1, 1), predicted, k=3))


    # make predictions for test data
    y_pred = clf.predict(X_test)
    predictions = [round(value) for value in y_pred]
    confusionMatrix = confusion_matrix(y_pred, y_test.astype(np.int64).tolist())
    #print confusionMatrix
    #plot_confusion_matrix(confusionMatrix)
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
