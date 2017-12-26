# -*- coding: utf-8 -*-
"""
Program For Naive Bayes Classifier using Python Library ScikitLearn
@author: Sourav Nandi; Project Work in ISI Kolkata in Dec 2017
DataSet url : https://archive.ics.uci.edu/ml/datasets/iris
"""

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

iris = datasets.load_iris() 
X = iris.data
y = iris.target

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=40)

gnb = GaussianNB()
mnb = MultinomialNB()

y_pred_gnb = gnb.fit(X_train, y_train).predict(X_test)
cnf_matrix_gnb = confusion_matrix(y_test, y_pred_gnb)
print('Confusion Matrix For Gaussian Naive Bayes Classifier: \n',cnf_matrix_gnb)

y_pred_mnb = mnb.fit(X_train, y_train).predict(X_test)
cnf_matrix_mnb = confusion_matrix(y_test, y_pred_mnb)
print('Confusion Matrix For Multinomial Naive Bayes Classifier: \n',cnf_matrix_mnb)

#Output:
"""
runfile('D:/ISI/Bayesian Classifier Proj/iris gaussian nb classifier.py', wdir='D:/ISI/Bayesian Classifier Proj')

Confusion Matrix For Gaussian Naive Bayes Classifier: 
 [[20  0  0]
 [ 0 20  0]
 [ 0  1 19]]
 
Confusion Matrix For Multinomial Naive Bayes Classifier: 
 [[20  0  0]
 [ 0 19  1]
 [ 0  1 19]]
"""