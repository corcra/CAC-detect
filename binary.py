#!/usr/bin/python
# author:       Stephanie Hyland (sh985@cornell.edu), but heavily based off the kNN and SVM notebooks at http://www.shogun-toolbox.org/page/documentation/notebook
# date:         March 2014
# description:  Does binary classification using kNN and SVM (Gaussian kernel).
# input:        Comma separated file, one row for each data 'point'.
#               Final field is binary label (-1,1).
# arguments:    filename, fraction of examples to use for training, K for kNN
# result:       Accuracy on the testing subset of the data for both methods.

import numpy as np
import sys
from numpy import random
from modshogun import *

if len(sys.argv)<4:
    sys.exit('USAGE: python binary.py data.csv train_frac num_nbh')

# Parameters (tune these!)
train_frac = float(sys.argv[2])
K=int(sys.argv[3])

# Prepare 'data frames'
Y_list = []
X_list = []

# Parse the data
datapath = sys.argv[1]
datafile = open(datapath,'r')
header = datafile.readline().strip().split(',')
for line in datafile:
    vals=line.strip().split(',')
    # assuming the final column is the label
    label=int(vals[-1])
    Y_list.append(label)
    features=map(float,vals[:-1])
    X_list.append(features)

print 'Number of features:',len(X_list[0])
# number of examples
n = len(Y_list)

# turn to numpy arrays
Y_array = np.array(Y_list,dtype=np.double)
# note transposition here!
X_array = np.array(X_list).transpose()

# keep this at a fixed number for reproducibility (if desired)
random.seed(0)

# Split into training and testing
subset = random.permutation(n)
training_indices = subset[:int(train_frac*n)]
testing_indices = subset[int(train_frac*n):]
print 'Number of training examples:',int(train_frac*n)
print 'Number of testing examples:',n-int(train_frac*n)

X_train = X_array[:,training_indices]
Y_train = Y_array[training_indices]

X_test = X_array[:,testing_indices]
Y_test = Y_array[testing_indices]

# Convert data to shogun objects
# MulticlassLabels must be 0,1 here
labels = MulticlassLabels((Y_train+1)/2)
bin_labels = BinaryLabels(Y_train)
features = RealFeatures(X_train)
labels_test = MulticlassLabels((Y_test+1)/2)
bin_labels_test = BinaryLabels(Y_test)
features_test = RealFeatures(X_test)

# kNN!
print '(kNN) k:',K
dist = EuclideanDistance()
knn = KNN(K,dist,labels)
knn.train(features)
knn_preds = knn.apply_multiclass(features_test)

# SVM!
# Why this width? Why this kernel? All good questions! They need answers.
print '(SVM) Using a Gaussian kernel.'
width = 2
kernel = GaussianKernel(features, features, width)
C = 1.0
svm = SVMLight(C, kernel, bin_labels)
_=svm.train()
svm_preds = svm.apply(features_test)

# Get accuracy
knn_evaluator = MulticlassAccuracy()
svm_evaluator = AccuracyMeasure()
knn_accuracy= knn_evaluator.evaluate(knn_preds,labels_test)
svm_accuracy= svm_evaluator.evaluate(svm_preds,bin_labels_test)
print "kNN Accuracy = %2.2f%%" % (100*knn_accuracy)
print "SVM Accuracy = %2.2f%%" % (100*svm_accuracy)
