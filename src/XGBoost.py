import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
import time
import ml_metrics as metrics
import matplotlib.pyplot as plt
import operator
from sklearn.metrics import confusion_matrix
import seaborn as sn



def plot_confusion_matrix(data):
    df_cm = pd.DataFrame(data, index=[i for i in range(10)],
                         columns=[i for i in range(10)])
                         plt.figure(figsize=(10, 7))
                         sn.heatmap(df_cm, annot=True)
                         plt.title("Confusion Matrix Obtained from XGBoost prediction")
                         plt.show()


def get_xgb_imp(xgb, feat_names):
    from numpy import array
    imp_vals = xgb.booster().get_fscore()
    imp_dict = {feat_names[i]:float(imp_vals.get('f'+str(i),0.)) for i in range(len(feat_names))}
    total = array(imp_dict.values()).sum()
    return {k:v/total for k,v in imp_dict.items()}


def plot(data,verbose):
    if verbose:
        sorted_x = sorted(data.items(), key=operator.itemgetter(1))
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



start = time.time()
inf = open("data_cleaned.csv")
feat_names= inf.readline().strip().split(',')
data = np.array([map(float, s.strip().split(',')) for s in inf.readlines()])

X = data[:, 0 : -1]
Y = data[:, -1 ]

print X.shape
print Y.shape


seed = 7

test_size = 0.3

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
print X_train.shape
print y_train.shape
print X_test.shape


temp = y_test.astype(np.int64).tolist()

xgdmat = xgb.DMatrix(X_train, y_train)

"""
    ######################## grid search ##################################
    cv = {'n_estimators': [200,400,600,800,1000] }
    params = {'learning_rate': 0.1,  'seed':0,  'colsample_bytree': 0.8,
    'max_depth': 5, 'min_child_weight': 1 }
    
    optimized_GBM = GridSearchCV(xgb.XGBClassifier(**params),
    cv,
    scoring = 'accuracy', cv = 3, n_jobs = -1)
    
    optimized_GBM.fit(X_train, y_train)
    var = optimized_GBM.grid_scores_
    print var
    print time.time() - start
    
    """

########################### final model #################################




params = {'learning_rate': 0.1,  'seed':0,  'colsample_bytree': 0.8, 'max_depth': 21,
    'min_child_weight': 1,'n_estimators': 100, 'subsample': 0.8, 'reg_lambda': 1 }
model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train)


feat_names =feat_names[:-1]
importance = get_xgb_imp(model, feat_names)
plot(importance, True)



# final_gb = xgb.train(params, xgdmat, num_boost_round = 500)
# xgb.plot_importance(model)
# plt.show()

preds = model.predict_proba(X_test)
predicted = preds.argsort(axis=1)[:, -np.arange(1, 6)]


# print "test Parmeter:"
# print i
print("score:",metrics.mapk(y_test.astype(np.int64).reshape(-1,1),predicted,k=3))


# testdmat = xgb.DMatrix(X_test)
# y_pred = final_gb.predict(testdmat)
y_pred = model.predict(X_test)

confusionMatrix = confusion_matrix(y_pred, y_test.astype(np.int64).tolist())
print confusionMatrix
plot_confusion_matrix(confusionMatrix)


predictions = [round(value) for value in y_pred]
#evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


print "time is:"
print time.time() - start
start = time.time()




# d_train = xgb.DMatrix(X_train, label=y_train)
# d_valid = xgb.DMatrix(X_test, label=y_test)
# watchlist = [(d_valid,'eval'),(d_train,'train')]
# clf = xgb.train(params, d_train, 200, watchlist, early_stopping_rounds=50, verbose_eval=True)


#
# xgdmat = xgb.DMatrix(X_train, y_train)

# final_gb = xgb.train(ind_params, xgdmat, num_boost_round = 432)


