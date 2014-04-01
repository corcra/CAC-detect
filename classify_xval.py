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
BINARY=True

# Parameters
alpha = 0.05
label_col = int(-2)

# Prepare 'data frames'
Y_list = []
X_list = []

# Parse the data
datafile = open(datapath,'rU')
header = datafile.readline().strip().split(',')
for line in datafile:
    vals=line.strip().split(',')
    if not 'inf' in vals:
        label=int(vals[label_col])
        # not interested in the final 5 columns
        features=map(float,vals[1:-5])
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

if verbose:
    # summarise settings, etc.
    print 'Number of features:',len(X_list[0])
    if verbose: print 'Total data size:',n
    # how many of each example do we have?
    for label in np.unique(Y_array):
        print 'Class:',int(label),':', sum(Y_array==label), 'examples'
    print 'Using '+str(cross_val)+'-fold cross-validation.'
print 'For kNN, K = ',K

knn_accuracy=[]
svm_accuracy=[]
features = RealFeatures(X_array)

# knn stuff
dist = EuclideanDistance()

# svm kernel details
width = 10
C = 1.0
d = 2
inhomo = True
size = 1
#kernel = GaussianKernel(features, features, width)
kernel = PolyKernel(features, features, d, inhomo)
#kernel = ExponentialKernel(features, features, width, dist, size)
#kernel = LinearKernel(features, features)
print 'For SVM, using',kernel.get_name()
#, 'width:',kernel.get_width()

MulticlassLabels(Y_array)
if BINARY:
    print 'Doing binary prediction!'
    svm_labels = BinaryLabels((-1)*(Y_array!=0)+(1)*(Y_array==0))
    knn_labels = MulticlassLabels(np.array(1*(Y_array!=0)+2*(Y_array==0),dtype=np.double))
  #  svm_metric = AccuracyMeasure()
    svm_metric = RecallMeasure()
  #  svm_metric = SpecificityMeasure()
    knn_metric = MulticlassAccuracy()
    svm = LibSVM(C,kernel,svm_labels)
    knn = KNN(K,dist,knn_labels)
    lock = False
else:
    print 'Doing multiclass prediction!'
    svm_labels = MulticlassLabels(Y_array)
    knn_labels = svm_labels
    svm_metric = MulticlassAccuracy()
    knn_metric = MulticlassAccuracy()
    svm = GMNPSVM(C,kernel,svm_labels)
    #svm = LibSVMMultiClass(C,kernel,svm_labels)
    knn = KNN(K,dist,knn_labels)
    lock = False

# Split strategy for cross-validation
if BINARY or cross_val<=9:
    svm_split = StratifiedCrossValidationSplitting(svm_labels,cross_val)
    knn_split = StratifiedCrossValidationSplitting(knn_labels,cross_val)
else:
    svm_split = CrossValidationSplitting(svm_labels,cross_val)
    knn_split = CrossValidationSplitting(knn_labels,cross_val)

# SVM
cross_svm = CrossValidation(svm, features, svm_labels, svm_split, svm_metric, lock)
cross_svm.set_num_runs(25)
cross_svm.set_conf_int_alpha(alpha)
result_svm=cross_svm.evaluate()
result_svm=CrossValidationResult.obtain_from_generic(result_svm)
print "SVM: (",svm_metric.get_name(),") %2.3f" % result_svm.mean, "[ %2.3f"  %result_svm.conf_int_low, "- %2.3f" % result_svm.conf_int_up,"]",str((1-alpha)*100),"% CI"
trained_svm = cross_svm.get_machine()


# kNN
cross_knn = CrossValidation(knn, features, knn_labels, knn_split, knn_metric, lock)
cross_knn.set_num_runs(25)
cross_knn.set_conf_int_alpha(alpha)
result_knn=cross_knn.evaluate()
result_knn=CrossValidationResult.obtain_from_generic(result_knn)
print "kNN: (",knn_metric.get_name(),") %2.3f" % result_knn.mean, "[ %2.3f"  %result_knn.conf_int_low, "- %2.3f" % result_knn.conf_int_up,"]",str((1-alpha)*100),"% CI"
trained_knn = cross_knn.get_machine()

# This particular analysis isn't great given I'm doing the cross-validation, I should just be converting the predicted specificity etc. into volumes.
# The problem with this is if the wrongly-predicted calcifications have a different volume distribution to the whole population, in which case the volume-based metric may be over or underestimated.
if BINARY:
    preds_svm = trained_svm.apply_binary(features)
    preds_knn = trained_knn.apply_multiclass(features)
   
    # Positive Indices
    true = np.where(Y_array==0)[0]
    false = np.where(Y_array!=0)[0]

    positive_svm = np.where(preds_svm[:]==1)[0]
    negative_svm = np.where(preds_svm[:]==(-1))[0]
    tp_svm = np.intersect1d(true,positive_svm)
    fp_svm = np.intersect1d(false,positive_svm)

    positive_knn = np.where(preds_knn[:]==2)[0]
    negative_knn = np.where(preds_knn[:]==1)[0]
    tp_knn = np.intersect1d(true,positive_knn)
    fp_knn = np.intersect1d(false,positive_knn)

    # Specify to volumes (to compare with Isgum et al)
    volumes = X_prenorm[:,0]
    true_volume = sum(volumes[true])
    false_volume = sum(volumes[false])

    tp_svm_volume = sum(volumes[tp_svm])
    fp_svm_volume = sum(volumes[fp_svm])

    tp_knn_volume = sum(volumes[tp_knn])
    fp_knn_volume = sum(volumes[fp_knn])

    TPR_svm_volume = tp_svm_volume/true_volume
    FPR_svm_volume = fp_svm_volume/false_volume
    
    TPR_knn_volume = tp_knn_volume/true_volume
    FPR_knn_volume = fp_knn_volume/false_volume
