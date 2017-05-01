import pandas as pd
from sklearn import svm
from sklearn import linear_model
import numpy as np
from heapq import nlargest
from sklearn import preprocessing

Data = pd.read_csv("../data/data_cleaned.csv")
s=np.random.permutation(Data.shape[0]);
Data=np.matrix(Data)
dim=Data.shape[0];

n=10;
recommend=3;
sample=np.random.permutation(dim); #print sample #[4,1,7,...6,2]\
accuracy_total=0;

#the function to calculate the prediction score
def acrcy(prob,true,recommend,classes):
	dim=prob.shape[0];
	#dim=3;
	sum=0;
	for i in range(dim):
		probs= prob[i].tolist();
		index= nlargest(recommend, range(prob.shape[1]), key=lambda i: probs[i])
		#print probs;
		#print index;
		#print true[i];
		for j in range(recommend):
			if(int(true[i])==int(classes[index[j]])):
				sum=sum+1/(1.0+j);
	acc=sum/dim;
	return acc;

# 10-fold cross validation
for i in range(n):
	if (i==n-1):
		test_index=sample[i*(dim/n):];
		train_index=sample[:i*(dim/n)];
	else:
		test_index=sample[i*(dim/n):(i+1)*(dim/n)];
		train_index=np.append(sample[:i*(dim/n)],sample[(i+1)*(dim/n):]);
	X=Data[train_index,:-1]; 
	y=Data[train_index,-1]; 	
	y=np.reshape(np.array(y), (len(train_index), ))
	#print X.shape, np.array(y).shape
	#print np.unique(y)
	Test=Data[test_index,:-1];
	true=np.array(Data[test_index,-1],dtype=int)


	# Use SVM to build the model
	'''
	#multiclass SVM
	clf = svm.SVC(decision_function_shape='ovr',probability=True,kernel='poly')
	clf.fit(X, y)  
	#pred_train=clf.predict(X)	
	#pred_test=clf.predict(Test)
	classes= clf.classes_
	prob_train=clf.predict_proba(X)
	prob_test=clf.predict_proba(Test)
	'''

	# Use logistic regression to build the model
	logreg = linear_model.LogisticRegression(multi_class='ovr',penalty ='l1',C=1);
	logreg.fit(X, y)
	pred_train=logreg.predict(X)
	classes=logreg.classes_
	prob_train=logreg.predict_proba(X)
	#pred_test=logreg.predict(Test)
	prob_test=logreg.predict_proba(Test)
	#print logreg.coef_ [0]
	
	accuracy=acrcy(prob_train,y,recommend,classes);
	print "train accuracy=",accuracy

	accuracy=acrcy(prob_test,true,recommend,classes);
	print "test accuracy=",accuracy	
	accuracy_total=accuracy_total+accuracy;


avg_acc=(accuracy_total+0.0)/n
print "average accuracy =",avg_acc












