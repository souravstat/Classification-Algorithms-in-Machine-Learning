# -*- coding: utf-8 -*-
"""
Program For Naive Bayes Classifier using Python
@author: Sourav Nandi
DataSet url : https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes
"""

# Example of Naive Bayes implemented from Scratch in Python
import csv
import random
import math

def OpenFile(filename):      #f1
	lines = csv.reader(open(filename, "rb"))
	dataset = list(lines)
	for i in range(len(dataset)):
		dataset[i] = [float(x) for x in dataset[i]]
	return dataset

def Create_Training_Set(dataset, splitRatio):      #f2
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]

def Divide_Classwise(dataset):       #f3
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
	return separated

def mean(numbers):                    #f4
	return sum(numbers)/float(len(numbers))

def stdev(numbers):                  #f5
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)

def Summary(dataset):     #f6
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	del summaries[-1]
	return summaries

def Classwise_Summary(dataset):       #f7
	separated = Divide_Classwise(dataset)      #call f3
	summaries = {}
	for classValue, instances in separated.iteritems():
		summaries[classValue] = Summary(instances)    #call f6
	return summaries

def calculateProbability(x, mean, stdev):       #f8
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def calculateClassProbabilities(summaries, inputVector):        #f9
	probabilities = {}
	for classValue, classSummaries in summaries.iteritems():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
	return probabilities
			
def predict(summaries, inputVector):        #f10
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.iteritems():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel

def getPredictions(summaries, testSet):     #f11
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions

def getAccuracy(testSet, predictions):      #f12
	correct = 0
	for i in range(len(testSet)):
		if testSet[i][-1] == predictions[i]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

def main():                                 #f13
	filename = 'pima-indians-diabetes.data.csv'
	splitRatio = 0.8
	dataset = OpenFile(filename)       #call f1
	trainingSet, testSet = Create_Training_Set(dataset, splitRatio)    #call f2
	print('Split {0} rows into train={1} and test={2} rows').format(len(dataset), len(trainingSet), len(testSet))
	# prepare model
	summaries = Classwise_Summary(trainingSet)     #Call f7
	# test model
	predictions = getPredictions(summaries, testSet)
	accuracy = getAccuracy(testSet, predictions)
	print('Accuracy: {0}%').format(accuracy)

main()

"""
Output:
    runfile('D:/ISI/Classification/pima indian/pima classifier - final.py', wdir='D:/ISI/Classification/pima indian')
Split 768 rows into train=614 and test=154 rows
Accuracy: 76.6753246753%

"""