# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 13:54:23 2017
Program For Naive Bayes Classifier using Python 3
Problem Description and Data: https://www.youtube.com/watch?v=ZAfarappAO0 (Francisco Iacobelli)
@author: Sourav Nandi
"""
import numpy as np
from collections import Counter, defaultdict

#We begin by calculating the prior probabilities of the two classes.
#We use a dictionary for this purpose
def prob_of_happening(list1):
    no_of_examples = len(list1)
    prob = dict(Counter(list1))
    for key in prob.keys():
        prob[key] = prob[key] / float(no_of_examples)
    return prob

def bayesian_classifier(training, outcome, new_sample):
    classes     = np.unique(outcome)
    rows, cols  = np.shape(training)
    likelihoods = {}
    for cls in classes:
        #initialise the dictionary
        likelihoods[cls] = defaultdict(list)
 
    class_probabilities = prob_of_happening(outcome)
 
    for cls in classes:
        row_indices = np.where(outcome == cls)[0]
        subset      = training[row_indices, :]
        r, c        = np.shape(subset)
        for j in range(0,c):
            likelihoods[cls][j] = list(subset[:,j])
 
   # for each cls in classes:
        for j in range(0,cols):
             likelihoods[cls][j] = prob_of_happening(likelihoods[cls][j])
 
 
    results = {}
    for cls in classes:
         class_probability = class_probabilities[cls]
         for i in range(0,len(new_sample)):
             relative_values = likelihoods[cls][i]
             if new_sample[i] in relative_values.keys():
                 class_probability *= relative_values[new_sample[i]]
             #else:
             #    class_probability *= 0
             results[cls] = class_probability
    print (results)
 
# Main Part Of Program
training   = np.asarray(((1,0,1,1),(1,1,0,0),(1,0,2,1),(0,1,1,1),
                         (0,0,0,0),(0,1,2,1),(0,1,2,0),(1,1,1,1)));
outcome    = np.asarray((0,1,1,1,0,1,0,1))
new_sample = np.asarray((1,0,1,0))
bayesian_classifier(training, outcome, new_sample)

#Demo with a toy example of different structure
training   = np.asarray(((1,0,1,1,1),(1,1,0,0,0),(1,0,2,1,0),(0,1,1,1,1),
                         (0,0,0,0,0),(1,0,1,2,1),(0,1,1,2,0)));
outcome    = np.asarray((0,1,1,1,0,1,0))
new_sample = np.asarray((1,0,1,0,0))
bayesian_classifier(training, outcome, new_sample)
