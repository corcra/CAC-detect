#!/usr/bin/python
# author:       Stephanie Hyland (sh985@cornell.edu), but heavily based off the kNN and SVM notebooks at http://www.shogun-toolbox.org/page/documentation/notebook
# date:         March 2014
# description:  Does multiclass/binary classification using kNN and SVM (Gaussian kernel).
# input:        Comma separated file, one row for each data 'point'.
#               Final field is binary label (-1,1).
# arguments:    filename, fraction of examples to use for training, K for kNN
# result:       Accuracy on the testing subset of the data for both methods.

import numpy as np
import sys
from numpy import random
from modshogun import *
from scipy import delete

if len(sys.argv)<4:
    sys.exit('USAGE: python classify.py data.csv train_frac K')

# Parameters (tune these!)
train_frac = float(sys.argv[2])
# what fold cross-validation?
cross_val = int(100)

# Option
verbose=False

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
# how many of each example do we have?
for label in np.unique(Y_array):
    print 'Class:',int(label),':', sum(Y_array==label), 'examples'

# centre and scale the data
X_prenorm = np.array(X_list)
X_trimmed = delete(X_prenorm,0,1)
# note transposition here!
X_array = X_trimmed.transpose()
print X_array.size


# keep this at a fixed number for reproducibility (if desired)
random.seed()

print 'Number of training examples:',int(train_frac*n)
print 'Number of testing examples:',n-int(train_frac*n)

knn_accuracy=[]
svm_accuracy=[]
for i in range(cross_val):
    # Split into training and testing
    subset = random.permutation(n)
    training_indices = subset[:int(train_frac*n)]
    testing_indices = subset[int(train_frac*n):]

    X_train = X_array[:,training_indices]
    Y_train = Y_array[training_indices]

    X_test = X_array[:,testing_indices]
    Y_test = Y_array[testing_indices]

    # Convert data to shogun objects
    # Multiclass, 0,1,2,3
    features = RealFeatures(X_train)
    features_test = RealFeatures(X_test)

    multi_labels = MulticlassLabels(Y_train)
    multi_labels_test = MulticlassLabels(Y_test)

    bin_labels = BinaryLabels(1*(Y_train!=0)+(-1)*(Y_train==0))
    bin_labels_test = BinaryLabels(1*(Y_test!=0)+(-1)*(Y_test==0))

    # kNN!
    if verbose: print "(kNN)",
    dist = EuclideanDistance()
    knn_evaluator = MulticlassAccuracy()
    knn_accuracies=[]
#    k_opts = range(20)
    k_opts = [int(sys.argv[3])]
    for K in k_opts:
        knn = KNN(K,dist,multi_labels)
        knn.train(features)
        knn_preds = knn.apply_multiclass(features_test)
        knn_acc= knn_evaluator.evaluate(knn_preds,multi_labels_test)
        knn_accuracies.append(knn_acc)

    best_accuracy=max(knn_accuracies)
    knn_accuracy.append(best_accuracy)
    best_K=[k_opts[i] for i in range(len(knn_accuracies)) if knn_accuracies[i]==best_accuracy]
    if verbose: print "Using K =","["+",".join(map(str,best_K))+"]","for best accuracy = %2.2f%%" % (100*best_accuracy)

    # SVM!
    # Why this width? Why this kernel? All good questions! They need answers.
    if verbose: print '(SVM) Using a Gaussian kernel.',
    width = 2
    kernel = GaussianKernel(features, features, width)
    C = 1.0
    svm = LibSVM(C, kernel, bin_labels)
    _=svm.train()
    svm_preds = svm.apply(features_test)
    svm_evaluator = AccuracyMeasure()
    svm_acc= svm_evaluator.evaluate(svm_preds,bin_labels_test)
    if verbose: print "Accuracy = %2.2f%%" % (100*svm_acc)
    svm_accuracy.append(svm_acc)

print 'kNN:', np.mean(knn_accuracy), np.sqrt(np.var(knn_accuracy))
print 'SVM:', np.mean(svm_accuracy), np.sqrt(np.var(svm_accuracy))
