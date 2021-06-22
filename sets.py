#!/usr/bin/env python
# coding: utf-8

#####################################################
# AI for CICY 4-folds                               #
#                                                   #
# Authors: H. Erbin, R. Finotello                   #
# Code: R. Finotello (riccardo.finotello@gmail.com) #
#                                                   #
#####################################################
#                                                   #
# Split training, validation and test sets.         #
#                                                   #
#####################################################

import os
import pandas as pd
import numpy as np
import argparse
import json

# set argument parser
parser = argparse.ArgumentParser()

parser.add_argument('-d', '--data', help='path to the JSON data file')
parser.add_argument('-tn', '--train', default=0.8, type=float, help='training ratio')
parser.add_argument('-vd', '--valid', default=0.1, type=float, help='validation ratio')
parser.add_argument('-tt', '--test', default=0.1, type=float, help='test ratio')
parser.add_argument('-s', '--stratified', type=str, help='name of the stratified variable')
parser.add_argument('-r', '--random', default=123, type=int, help='random seed')
parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')

args = parser.parse_args()

# load the dataset
df = pd.read_json(args.data, orient='index')

# divide into training and test sets
train = None
valid = None
test  = None

if args.stratified is None:

    train = df.sample(frac=args.train, random_state=args.random)
    oos   = df.loc[~df.index.isin(train.index), :]
    
    valid = oos.sample(frac=round(args.valid / (1.0 - args.train), 2))
    oos   = oos.loc[~oos.index.isin(valid.index), :]
    
    test  = oos.sample(frac=round(args.test / (1.0 - args.train - args.test), 2))
    
else:

    obj = df.groupby(by=args.stratified, group_keys=False)
    
    train = obj.apply(lambda x: x.sample(frac=args.train, random_state=args.random))
    oos   = df.loc[~df.index.isin(train.index), :]
    
    obj = oos.groupby(by=args.stratified, group_keys=False)
    
    valid = obj.apply(lambda x: x.sample(frac=round(args.valid / (1.0 - args.train), 2)))
    oos   = oos.loc[~oos.index.isin(valid.index), :]
    
    test  = oos.sample(frac=round(args.test / (1.0 - args.train - args.test), 2))
    
if args.verbose:
    print(f'Training set:   {train.shape[0]:d} rows ({100 * train.shape[0] / df.shape[0]:.1f}% ratio)')
    print(f'Validation set: {valid.shape[0]:d} rows ({100 * valid.shape[0] / df.shape[0]:.1f}% ratio)')
    print(f'Test set:       {test.shape[0]:d} rows ({100 * test.shape[0] / df.shape[0]:.1f}% ratio)')
    
# divide into features and labels
feat = 'matrix'
labs = ['h11', 'h21', 'h31', 'h22']

X_train = {feat: list(train[feat].values)}
y_train = {l: list(train[l].astype(np.float).values.reshape(-1,)) for l in labs}

with open(f'X_train_{int(100*args.train):d}.json', 'w') as f:
    json.dump(X_train, f)
with open(f'y_train_{int(100*args.train):d}.json', 'w') as f:
    json.dump(y_train, f)

X_valid = {feat: list(valid[feat].values)}
y_valid = {l: list(valid[l].astype(np.float).values.reshape(-1,)) for l in labs}

with open(f'X_valid_{int(100*args.valid):d}.json', 'w') as f:
    json.dump(X_valid, f)
with open(f'y_valid_{int(100*args.valid):d}.json', 'w') as f:
    json.dump(y_valid, f)

X_test = {feat: list(test[feat].values)}
y_test = {l: list(test[l].astype(np.float).values.reshape(-1,)) for l in labs}

with open(f'X_test_{int(100*args.test):d}.json', 'w') as f:
    json.dump(X_test, f)
with open(f'y_test_{int(100*args.test):d}.json', 'w') as f:
    json.dump(y_test, f)

X_test_full = {feat: list(oos[feat].values)}
y_test_full = {l: list(oos[l].astype(np.float).values.reshape(-1,)) for l in labs}

with open(f'X_test_full_{int(100*args.train):d}.json', 'w') as f:
    json.dump(X_test_full, f)
with open(f'y_test_full_{int(100*args.train):d}.json', 'w') as f:
    json.dump(y_test_full, f)

