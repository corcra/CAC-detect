#!/usr/bin/python
# author:       Stephanie Hyland (sh985@cornell.edu), but heavily based off the kNN and SVM notebooks at http://www.shogun-toolbox.org/page/documentation/notebook
# date:         March 2014
# description:  Does multiclass/binary classification using kNN and SVM (Gaussian kernel), using stratified cross-validation.
# input:        Comma separated file, one row for each data 'point'.
#               Final field is class label (integer).
# arguments:    filename, what-fold validation?, K for kNN
# result:       Accuracy on the testing subset of the data for both methods.

import numpy as np
import sys
from numpy import random
from modshogun import *

if len(sys.argv)<4:
    sys.exit('USAGE: python classify.py data.csv nfold K')

# Inputs
datapath = sys.argv[1]
cross_val = int(sys.argv[2])
K = int(sys.argv[3])

# Options
verbose=False
binary=True

# Parameters
alpha = 0.05

# Prepare 'data frames'
Y_list = []
X_list = []

# Parse the data
datafile = open(datapath,'r')
header = datafile.readline().strip().split(',')
for line in datafile:
    vals=line.strip().split(',')
    # assuming the final column is the label
    if not 'inf' in vals:
        label=int(vals[-1])
        features=map(float,vals[1:-1])
        X_list.append(features)
        Y_list.append(label)

# number of examples
n = len(Y_list)

# turn to numpy arrays
Y_array = np.array(Y_list,dtype=np.double)
X_prenorm = np.array(X_list)

# normalise (subtracting mean, dividing by s.d.)
X_normalised = (X_prenorm - np.mean(X_prenorm,0))/np.sqrt(np.var(X_prenorm,0))
# transpose (shogun expects this format)
X_array = X_normalised.transpose()

# keep this at a fixed number for reproducibility (if desired)
random.seed()

# summarise settings, etc.
print 'Number of features:',len(X_list[0])
print 'Total data size:',n
# how many of each example do we have?
for label in np.unique(Y_array):
    print 'Class:',int(label),':', sum(Y_array==label), 'examples'
print 'Using '+str(cross_val)+'-fold cross-validation.'
print 'For kNN, K = ',K,'\n'

knn_accuracy=[]
svm_accuracy=[]
features = RealFeatures(X_array)

# svm kernel details
width = 2
kernel = GaussianKernel(features, features, width)
C = 1.0

# knn stuff
dist = EuclideanDistance()

MulticlassLabels(Y_array)
if binary:
    print 'Doing binary prediction!'
    svm_labels = BinaryLabels(1*(Y_array!=0)+(-1)*(Y_array==0))
    knn_labels = MulticlassLabels(np.array(1*(Y_array!=0)+2*(Y_array==0),dtype=np.double))
#    metric = AccuracyMeasure()
    svm_metric = PrecisionMeasure()
    knn_metric = MulticlassAccuracy()
    svm = LibSVM(C,kernel,svm_labels)
    knn = KNN(K,dist,knn_labels)
    lock = False
else:
    print 'Doing multiclass prediction!'
    labels = MulticlassLabels(Y_array)
    knn_labels = labels
    svm_metric = MulticlassAccuracy()
    knn_metric = MulticlassAccuracy()
    svm = GMNPSVM(C,kernel,svm_labels)
    knn = KNN(K,dist,knn_labels)
    lock = False

# Split strategy for cross-validation
svm_split = StratifiedCrossValidationSplitting(svm_labels,cross_val)
knn_split = StratifiedCrossValidationSplitting(knn_labels,cross_val)

# SVM
cross_svm = CrossValidation(svm, features, svm_labels, svm_split, svm_metric, lock)
cross_svm.set_num_runs(25)
cross_svm.set_conf_int_alpha(alpha)
result_svm=cross_svm.evaluate()
result_svm=CrossValidationResult.obtain_from_generic(result_svm)
print "SVM: (",svm_metric.get_name(),") %2.3f" % result_svm.conf_int_low, "[ %2.3f"  %result_svm.mean, "- %2.3f" % result_svm.conf_int_up,"]",str((1-alpha)*100),"% CI"

# kNN
cross_knn = CrossValidation(knn, features, knn_labels, knn_split, knn_metric, lock)
cross_knn.set_num_runs(25)
cross_knn.set_conf_int_alpha(alpha)
result_knn=cross_knn.evaluate()
result_knn=CrossValidationResult.obtain_from_generic(result_knn)
print "kNN: (",knn_metric.get_name(),") %2.3f" % result_knn.conf_int_low, "[ %2.3f"  %result_knn.mean, "- %2.3f" % result_knn.conf_int_up,"]",str((1-alpha)*100),"% CI"
