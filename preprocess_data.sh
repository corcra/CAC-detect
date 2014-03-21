#!/bin/bash

pos_cac=pos_train.csv
neg_ac=neg_train_nac.csv
neg_mc=neg_train_nmc.csv
neg_oc=neg_train_noc.csv

# sed bit is to strip the header line
sed '1d' $pos_cac | awk 'BEGIN{FS=","}{ print $0","0 }' > cac.temp
sed '1d' $neg_ac | awk 'BEGIN{FS=","}{ print $0","1 }' > ac.temp
sed '1d' $neg_mc | awk 'BEGIN{FS=","}{ print $0","2 }' > mc.temp
sed '1d' $neg_oc | awk 'BEGIN{FS=","}{ print $0","3 }' > oc.temp

# stick together
cat cac.temp ac.temp mc.temp oc.temp > training_all.csv

# tidy up
rm -v *.temp
