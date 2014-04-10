#!/bin/python
# author:       Stephanie Hyland (sh985@cornell.edu)
# date:         April 2014
# description:  Binary classification for CACs using a conditional random field (CRF).
#               A graph consists of all calcifications from a single patient.
#               Calcifications are connected in the graph if within "radius" of each other.
# input:        Comma-separated file, one row per calcification, with patient IDs and labels.
# result:       Performance evaluation on a test/train split of the input data.
#
# todo:         1. n-fold cross-validation.
#               2. proper treatment of features

import pystruct
from pystruct.models import GraphCRF
from pystruct.learners import OneSlackSSVM
import sys
import numpy as np
import random
import math

# --- Parameters --- #
# the patient id is the final column
id_col = -1
# label is the second last column
label_col = -2
# any other indices to drop? (this is dataspecific...)
feature_indices = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
# spatial indices, for calculating distances (we should be losing these anyway)
space_indices = [16,17,18]
# what features to use? (select this!)
n_features = len(feature_indices)
# how many states (2)
n_states = 2
# what fraction of the data to use for testing?
train_frac = 0.9
# how much output to give
verbose=True

# --- Inputs --- #
if len(sys.argv)<3:
    sys.exit("Datafile and radius please.")

# datafile
datapath=sys.argv[1]
# radius for 'neighbourness'? (atm totally arbitrary)
radius = sys.argv[2]

# --- Functions --- #
def get_neighbours(X):
    n_node = X.shape[0]
    nbs = []
    for n in xrange(n_node):
        for nn in xrange(n+1,n_node):
            distance = np.linalg.norm(X[n,space_indices]-X[nn,space_indices])
            if distance<radius:
                nbs.append((n,nn))
    return nbs

def prepare_data(datafile):
    header = datafile.readline()
    data = dict()
    for line in datafile:
        splitline = line.strip().split(',')
        # each patient is a graph example
        patient_id = splitline[id_col]
        try:
            data[patient_id].append(splitline)
        except KeyError:
            data[patient_id] = [splitline]

    examples = []
    labels = []
    for patient in data:
        all_features= np.array(data[patient],dtype=float)
        edges = np.array(get_neighbours(all_features))
        patient_data = (all_features[:,feature_indices],edges)
        if edges.shape[0]>0:
            examples.append(patient_data)
            labels.append(1*(all_features[:,label_col]==0).astype(np.int32))
   
    return examples, labels

def get_contingency(pred,true):
    pos = np.where(true==1)
    neg = np.where(true==0)
    TP = sum(pred[pos]==1)
    FP = sum(pred[neg]==1)
    TN = sum(pred[neg]==0)
    FN = sum(pred[pos]==0)
    return np.array([TP,FP,TN,FN])

# --- Data prep --- #
datafile = open(datapath,'rU')
examples, labels = prepare_data(datafile)
n_examples = len(examples)

# --- Test/train split --- #
train_indices = random.sample(range(n_examples),int(math.ceil(train_frac*n_examples)))
X_train = [examples[i] for i in train_indices]
Y_train = [labels[i] for i in train_indices]

test_indices = [i for i in range(n_examples) if not i in train_indices]
X_test = [examples[i] for i in test_indices]
Y_test = [labels[i] for i in test_indices]

if verbose:
    print "n_examples",n_examples
    print "n_train:",len(train_indices)
    print "n_test:",len(test_indices)

# --- Train model --- #
model = GraphCRF(n_states,n_features)
ssvm = OneSlackSSVM(model=model, C=.1, inference_cache=50, tol=0.1, verbose=0)
ssvm.fit(X_train, Y_train)

# --- Test with pystruct --- #
print("Test score with graph CRF: %f" % ssvm.score(X_test, Y_test))

# --- Test manually - get contingency tables --- #
prediction = ssvm.predict(X_test)

contingency = np.array([0,0,0,0])
for i in xrange(len(test_indices)):
    pred = prediction[i]
    true = Y_test[i]
    contingency = contingency+get_contingency(pred,true)

TP, FP, TN, FN = contingency[0], contingency[1], contingency[2], contingency[3]

sens = float(TP)/(TP+FN)
spec = float(TN)/(FP+TN)

print("Sensitivity: %f" % sens)
print("Specificity: %f" % spec)
print "Contingency table: (TP FP TN FN):", contingency
