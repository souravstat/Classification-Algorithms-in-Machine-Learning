# -*- coding: utf-8 -*-
from __future__ import division
import json
import pickle
import random
import math
import operator
from collections import OrderedDict


# Import vectors previusly getted from MongoDB
vectors = pickle.load(open("data/vectors.p", "rb"))

# Shuffle the dict keys
rand_keys = []
keys = list(vectors.keys())
i = 0
while len(keys) > 0:
    element = random.choice(keys)
    keys.remove(element)
    rand_keys.append(element)
    i += 1

# Parameters
k = 40
min_limit = 0
max_limit = min_limit + k
errors = []

# Exceute k-fold test
while max_limit <= len(vectors):
    
    # DS to manage information
    vectors_to_testing = {}
    entity_frecuency = {}
    cont_training = 0
    cont_sample = 0
    cont_total = 0
    
    # Run k-fold testing
    for id in rand_keys:
        # Sampling
        if cont_total >= min_limit and cont_total < max_limit:
            cont_sample += 1
            vectors_to_testing[cont_sample] = {}
            vectors_to_testing[cont_sample]["supervised"] = vectors[id]["supervised_tags"]
            i = 0
            while i < len(vectors_to_testing[cont_sample]["supervised"]):
                vectors_to_testing[cont_sample]["supervised"][i] = vectors_to_testing[cont_sample]["supervised"][i].lower()
                i += 1
            vectors_to_testing[cont_sample]["text"] = vectors[id]["text"][0].lower()+" "+vectors[id]["text"][1].lower()
            vectors_to_testing[cont_sample]["raw_tags"] = vectors[id]["raw_tags"]
            i = 0
            while i < len(vectors_to_testing[cont_sample]["raw_tags"]):
                vectors_to_testing[cont_sample]["raw_tags"][i] = vectors_to_testing[cont_sample]["raw_tags"][i][0].lower()
                i += 1
        else:
            # Update model with training vectors
            for entity in vectors[id]["supervised_tags"]:
                if entity not in entity_frecuency:
                    entity_frecuency[entity] = 1
                else:
                    entity_frecuency[entity] += 1
            cont_training += 1
        cont_total += 1
    #print("Training vectors: %s" % cont_training)   
    #print("Sampling vectors: %s" % cont_sample) 
    
    
    # Tacking into account the tail distribution
    for index in vectors_to_testing:
        head_dist_entities = {}
        for entity in vectors_to_testing[index]["raw_tags"]:
            if entity in entity_frecuency:
                head_dist_entities[entity] = entity_frecuency[entity]
            else:
                head_dist_entities[entity] = 0
                
        aux = []
        items = sorted(head_dist_entities.items(), key=lambda x: x[1], reverse = True)
        cont = 0
        for item in items:
            aux.append(item[0])
            if cont > 40:
                break
            cont += 1
        vectors_to_testing[index]["raw_tags"] = aux
        
    # Start to check the vectors
    error = []
    for id in vectors_to_testing:
        entities = vectors_to_testing[id]["supervised"]
        raw_text = vectors_to_testing[id]["text"]
        n = len(entities)
    
        error_entity = []
        new_entity = []
        #print("-->",vectors_to_testing[id]["raw_tags"],"\n")
        for entity in entities:
            if raw_text.find(entity) == -1 and entity not in vectors_to_testing[id]["raw_tags"]:
                new_entity.append(entity)
            elif raw_text.find(entity) != -1 and entity not in vectors_to_testing[id]["raw_tags"]:
                error_entity.append(entity)
        
        """
        print("NER:%s\n" % vectors_to_testing[id]["raw_tags"])
        print("Supervised:%s\n" % vectors_to_testing[id]["supervised"])
        print("Errors:%s %s" % (len(error_entity), error_entity))
        print("New:%s %s\n" % (len(new_entity), new_entity))
        print(vectors_to_testing[id]["text"],"\n")
        """
        n -= len(new_entity)
        if len(error_entity) == 0:
            ratio = 0
        else:
            ratio = (len(error_entity)/n)
        ratio = 1-ratio
        if len(entities) > 0:
            error.append(ratio)
        
    #print(sum(error)/float(cont_sample))
    errors.append(sum(error)/float(len(error)))
    
    # Update k-fold iteration
    min_limit += k
    max_limit += k
    
# Calculate mean and sd from errors
n = float(len(errors))
mean = sum(errors) / n
deviation = 0
for error in errors:
    deviation += pow((float(error - mean)), 2)
std_deviation = math.sqrt(deviation) * 1/n
print("Mean: %f\tDS:%f" % (mean, std_deviation))


    
    
    
    