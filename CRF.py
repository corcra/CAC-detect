#!/bin/python
# author:       Stephanie Hyland (sh985@cornell.edu), based off factor graph notebook at http://www.shogun-toolbox.org/static/notebook/current/FGM.html
# date:         April 2014
# description:  Does binary classification for CACs using a factor graph.
# input:        Comma separated file, one row for each calcification.
# result:       A CRF model trained to identify CACs, plus measures of its performance.

# notes:        The factor graph consists of a series of nodes, each with possible values of 0 or 1 (CAC or not). We want to jointly predict the whole collection for a patient.
#               The first thing we have to do is construct the graph.

import numpy as np
from modshogun import *
import time
import sys
import math
import random

# --- Parameters --- #
# the patient id is the final column
id_col = -1
# any other indices to drop? (this is dataspecific...)
feature_indices = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
# spatial indices, for calculating distances (we should be losing these anyway)
space_indices = [16,17,18]
# what features to use? (select this!)
n_features = len(feature_indices)
# how many states (2)
n_states = 2
# what fraction of the data to use for testing?
test_frac = 0.1
# radius for 'neighbourness'? (atm totally arbitrary)
radius = 30
# inference method?
inference=TREE_MAX_PROD

def get_contingency(pred,true):
    pos = np.where(true==1)
    neg = np.where(true==0)
    TP = sum(pred[pos]==1)
    FP = sum(pred[neg]==1)
    TN = sum(pred[neg]==0)
    FN = sum(pred[pos]==0)
    return np.array([TP,FP,TN,FN])

def get_labels(X):
    # labels are the last column
    n_calc = X.shape[0]
    Y = 1*(X[:,-2]==0).astype(np.int32)
    loss_weights = np.array([1.0/n_calc]*n_calc)
    graph_obs = FactorGraphObservation(Y, loss_weights)
    return graph_obs

def get_neighbours(X):
    n_node = X.shape[0]
    nbs = []
    for n in xrange(n_node):
        for nn in xrange(n+1,n_node):
            distance = np.linalg.norm(X[n,space_indices]-X[nn,space_indices])
            if distance<radius:
                print 'neighbours!', n, nn, distance
                nbs.append((n,nn))
    return nbs

def get_features(fg,X,ftypes):
    # input is a list of calcifications with their features (excluding label)
    # question: normalisation of this...? can't really locally whiten this
    n_calc = X.shape[0]
    # the last 'var' is actually the label

    # unary
    for c in xrange(n_calc):
#        dat_unary = np.array(1*(X[c,feature_indices]>0.5),dtype=float)
        dat_unary = X[c,feature_indices]
        calc_index_unary = np.array([c],np.int32)
        fac_unary = Factor(ftypes[0],calc_index_unary,dat_unary)
        fg.add_factor(fac_unary)

    # pairwise
#    for pair in get_neighbours(X):
#        dat_p = np.array([1.0])
#        print pair
#        calc_index_p = np.array(pair,np.int32)
#        fac_p = Factor(ftypes[1],calc_index_p,dat_p)
#        fg.add_factor(fac_p)

    # pairwise, adjacent
    for cc in xrange(n_calc-1):
        dat_p = np.array([1.0])
        calc_index_p = np.array([cc,cc+1],np.int32)
        fac_p = Factor(ftypes[1],calc_index_p,dat_p)
        fg.add_factor(fac_p)

    # "bias" (i just introduced this to see if it'd fix my problem)
#    dat_bias = np.array([1.0])
#    calc_index_bias = np.array([0],np.int32)
#    fac_bias = Factor(ftypes[2],calc_index_bias,dat_bias)
#    fg.add_factor(fac_bias)

    #print 'calcs:',n_calc
    #print 'edges:',fg.get_num_edges()
    #print 'factors:',fg.get_num_factors()
#    # uncomment for errors...
#    print 'connected?', fg.is_connected_graph()
#    print 'tree graph?',fg.is_tree_graph()
    return fg

def get_ftypes(n_features):
    # unary
    card_unary = np.array([n_states],np.int32)
    weights_unary = np.zeros(n_states*n_features)
    fac_unary = TableFactorType(0,card_unary,weights_unary)

    # pairwise
    card_pair = np.array([n_states,n_states],np.int32)
    weights_pair = np.zeros(n_states*n_states)
    fac_pair = TableFactorType(1,card_pair,weights_pair)

    # "bias"
#    card_bias = np.array([n_states],np.int32)
#    weights_bias = np.zeros(n_states)
#    fac_bias = TableFactorType(2,card_bias,weights_bias)

    # all the params
#    weights_initial = [weights_unary,weights_pair,weights_bias]
    weights_initial = [weights_unary,weights_pair]

    # all the factors
#    ftypes = [fac_unary, fac_pair,fac_bias]
    ftypes = [fac_unary, fac_pair]
    return ftypes

def get_samples_labels(data,ftypes):
    num_patients = len(data)
    samples = FactorGraphFeatures(num_patients)
    labels = FactorGraphLabels(num_patients)

    for patient in data:
        patient_data = np.array(data[patient],dtype=float)
        # i THINK this might be a reference to VC dimension
        n_calc = len(patient_data)
        VC = np.array([n_states]*n_calc,np.int32)

        patient_fg = FactorGraph(VC)

        patient_fg = get_features(patient_fg, patient_data,ftypes)
        patient_labels = get_labels(patient_data)

        samples.add_sample(patient_fg)
        labels.add_label(patient_labels)

    return samples, labels

def parse_data(datafile):
    # input is the raw CSV file
    # output is a dictionary: one element for each patient
    # within each patient, we have one line for each CAC
    # within each CAC, we have the raw information
    header = datafile.readline()
    data = dict()
    for line in datafile:
        splitline = line.strip().split(',')
        patient_id = splitline[id_col]
        try:
            data[patient_id].append(splitline)
        except KeyError:
            data[patient_id] = [splitline]
    return data 

# --- end of functions --- #

# get the data
datapath = sys.argv[1]
datafile = open(datapath,'rU')
data = parse_data(datafile)

# create test/train split
patient_ids = data.keys()
n_test = int(math.ceil(test_frac*len(patient_ids)))
n_train = len(patient_ids)-n_test
test_ids = random.sample(patient_ids,n_test)

test_data = dict()
train_data = dict()
for patient in data:
    if patient in test_ids:
        test_data[patient] = data[patient]
    else:
        train_data[patient] = data[patient]

# --- Create model --- #
ftypes = get_ftypes(n_features)
train_samples, train_labels = get_samples_labels(train_data,ftypes)
model = FactorGraphModel(train_samples, train_labels, inference)
for ftype in ftypes:
    model.add_factor_type(ftype)

# --- Training --- #
# bundle method for regularized risk minmisation (lambda = 0.01)
bmrm = DualLibQPBMSOSVM(model, train_labels, 0.01)

# absolute tolerance (parameter)
bmrm.set_TolAbs(20.0)
bmrm.set_verbose(True)

print "Training on", n_train, "patients, with",sum(map(len,train_data.values())),"calcifications."

t0 = time.time()
bmrm.train()
t1 = time.time()

weights_bmrm = bmrm.get_w()

print "Training took", t1-t0,'seconds.'

# --- Testing --- #
test_samples, test_labels = get_samples_labels(test_data,ftypes)

TP, FP, TN, FN = 0, 0, 0, 0
contingency = np.array([TP,FP,TN,FN])

for ts in xrange(n_test):
    fg_test = test_samples.get_sample(ts)
    fg_test.compute_energies()
    fg_test.connect_components()

    infer_met = MAPInference(fg_test, inference)
    infer_met.inference()

    y_pred = infer_met.get_structured_outputs()
    y_truth = FactorGraphObservation.obtain_from_generic(test_labels.get_label(ts))
    contingency = contingency + get_contingency(np.array(y_pred.get_data()),np.array(y_truth.get_data()))

TP, FP, TN, FN = contingency[0], contingency[1], contingency[2], contingency[3]

print contingency
sens = float(TP)/(TP+FN)
spec = float(TN)/(FP+TN)

print "Testing on",n_test,"patients, with",sum(contingency),"calcifications."
print("Sensitivity: %.4f" % sens)
print("Specificity: %.4f" % spec)

# From here on it's just basically a direct copy from the notebook, I should work through this myself as well.
# training error of BMRM method
bmrm.set_w(weights_bmrm)
model.w_to_fparams(weights_bmrm)
lbs_bmrm = bmrm.apply()
acc_loss = 0.0
ave_loss = 0.0
for i in xrange(n_train):
    y_pred = lbs_bmrm.get_label(i)
    y_truth = train_labels.get_label(i)
    acc_loss = acc_loss + model.delta_loss(y_truth, y_pred)
ave_loss = acc_loss / n_train
print('BMRM: Average training error is %.4f' % ave_loss)

# testing error
bmrm.set_features(test_samples)
bmrm.set_labels(test_labels)
lbs_bmrm_ts = bmrm.apply()
acc_loss = 0.0
ave_loss_ts = 0.0

for i in xrange(n_test):
    y_pred = lbs_bmrm_ts.get_label(i)
    y_truth = test_labels.get_label(i)
    acc_loss = acc_loss + model.delta_loss(y_truth, y_pred)
ave_loss_ts = acc_loss / n_test
print('BMRM: Average testing error is %.4f' % ave_loss_ts)
