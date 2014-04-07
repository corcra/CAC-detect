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

# the patient id is the final column
id_col = -1
# any other indices to drop? (this is dataspecific...)
lose_indices = [16,17,18,20]
# what features to use? (select this!)
n_features = 16
# how many states (2)
n_states = 2

def get_labels(X):
    # labels are the last column
    n_calc = X.shape[0]
    n_vars = X.shape[1]-1
    Y = X[:,-1].astype(np.int32)
    loss_weights = np.array([1.0/n_calc]*n_calc)
    graph_obs = FactorGraphObservation(Y, loss_weights)
    return graph_obs

def get_features(fg,X,ftypes):
    # input is a list of calcifications with their features (excluding label)
    # question: normalisation of this...? can't really locally whiten this
    n_calc = X.shape[0]
    # the last 'var' is actually the label
    n_vars = X.shape[1]-1

    # unary
    for c in xrange(n_calc):
        dat_unary = X[c,:]
        calc_index_unary = np.array([c],np.int32)
        fac_unary = Factor(ftypes[0],calc_index_unary,dat_unary)
        fg.add_factor(fac_unary)

    # pairwise (for now, arbitrary/random/weird)
    for cc in xrange(n_calc-1):
        dat_p = np.array([1.0])
        calc_index_p = np.array([cc,cc+1],np.int32)
        fac_p = Factor(ftypes[1],calc_index_p,dat_p)
        fg.add_factor(fac_p)

    # "bias" (i just introduced this to see if it'd fix my problem)
    dat_bias = np.array([1.0])
    calc_index_bias = np.array([0],np.int32)
    fac_bias = Factor(ftypes[2],calc_index_bias,dat_bias)
    fg.add_factor(fac_bias)

    print 'calcs:',n_calc
    print 'vars:',n_vars
    print 'edges:',fg.get_num_edges()
    print 'factors:',fg.get_num_factors()
    print 'acyclic?', fg.is_acyclic_graph()
    print 'connected?', fg.is_connected_graph()
    print 'tree graph?',fg.is_tree_graph()
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
    card_bias = np.array([n_states],np.int32)
    weights_bias = np.zeros(n_states)
    fac_bias = TableFactorType(2,card_bias,weights_bias)

    # all the params
    weights_initial = [weights_unary,weights_pair,weights_bias]

    # all the factors
    ftypes = [fac_unary, fac_pair,fac_bias]
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
    keep_indices = [i for i in range(21) if not i in lose_indices]
    for line in datafile:
        splitline = line.strip().split(',')
        patient_id = splitline[id_col]
        try:
            prep_line = [splitline[i] for i in keep_indices]
            data[patient_id].append(prep_line)
        except KeyError:
            prep_line = [splitline[i] for i in keep_indices]
            data[patient_id] = [prep_line]
    return data 

# --- end of functions --- #

# get the data
datapath = sys.argv[1]
datafile = open(datapath,'rU')
data = parse_data(datafile)

# --- Create model --- #
ftypes = get_ftypes(n_features)
samples, labels = get_samples_labels(data,ftypes)
model = FactorGraphModel(samples, labels, TREE_MAX_PROD)
for ftype in ftypes:
    model.add_factor_type(ftype)

# --- Training --- #
# bundle method for regularized risk minmisation (lambda = 0.01)
bmrm = DualLibQPBMSOSVM(model, labels, 0.01)

# absolute tolerance (parameter)
bmrm.set_TolAbs(20.0)
bmrm.set_verbose(True)

t0 = time.time()
bmrm.train()
t1 = time.time()

weights_bmrm = bmrm.get_w()

print 'Took', t1-t0,'seconds.'
