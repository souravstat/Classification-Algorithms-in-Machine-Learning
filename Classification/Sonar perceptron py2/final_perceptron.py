# -*- coding: utf-8 -*-
"""
Program For Perceptron Algorithm using Python.
@author: Sourav Nandi; Project Work in ISI Kolkata
DataSet url : https://http://archive.ics.uci.edu/ml/datasets/connectionist+bench+(sonar,+mines+vs.+rocks)
"""

from random import seed
from random import randrange
from csv import reader
 
# Load a CSV file
def open_file(filename):									#f1
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset
 
# Convert string column to float
def convert_to_float(dataset, column):				#f2
	for row in dataset:
		row[column] = float(row[column].strip())
 
# Convert string column to integer
def convert_to_int(dataset, column):					#f3
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup
 
# Split a dataset into k folds
def k_fold_cross_validation(dataset, n_folds):			#f4
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split
 
# Calculate accuracy percentage
def accuracy_metric(actual, predicted):					#f5
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0
 
# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):		#f6
	folds = k_fold_cross_validation(dataset, n_folds)		#call f4
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)	
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)		#call f5
		scores.append(accuracy)
	return scores
 
# Make a prediction with weights
def predict(row, weights):								#f7
	activation = weights[0]
	for i in range(len(row)-1):
		activation += weights[i + 1] * row[i]
	return 1.0 if activation >= 0.0 else 0.0
 
# Estimate Perceptron weights using stochastic gradient descent
def train_weights(train, l_rate, n_epoch):				#f8
	weights = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		for row in train:
			prediction = predict(row, weights)			#call f7
			error = row[-1] - prediction
			weights[0] = weights[0] + l_rate * error
			for i in range(len(row)-1):
				weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
	return weights
 
# Perceptron Algorithm With Stochastic Gradient Descent
def perceptron(train, test, l_rate, n_epoch):			#f9
	predictions = list()
	weights = train_weights(train, l_rate, n_epoch)		#call f8
	for row in test:
		prediction = predict(row, weights)				#call f7
		predictions.append(prediction)
	return(predictions)
 
# Test the Perceptron algorithm on the sonar dataset
seed(1)
# load and prepare data
filename = 'sonar.all-data.csv'
dataset = open_file(filename)							#call f1
for i in range(len(dataset[0])-1):
	convert_to_float(dataset, i)						#call f2
# convert string class to integers
convert_to_int(dataset, len(dataset[0])-1)			#call f3
# evaluate algorithm
n_folds = 4
l_rate = 0.05
n_epoch = 500
scores = evaluate_algorithm(dataset, perceptron, n_folds, l_rate, n_epoch)		#call f6
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

"""
Output:
Scores: [80.76923076923077, 82.6923076923077, 73.07692307692307, 71.15384615384616]
Mean Accuracy: 76.923%

"""