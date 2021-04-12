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
# Create the hyperparameter list.                   #
#                                                   #
#####################################################


import numpy as np
from sklearn.model_selection import ParameterGrid
import argparse
import json

# set argument parser
parser = argparse.ArgumentParser()

parser.add_argument('-p', '--params', default='./parameters.json', help='path to the model parameters')
parser.add_argument('-o', '--out', default='./hyper_list.json', help='path to the list of hyperparameters')
parser.add_argument('-s', '--search', default=10, type=int, help='no. of points in search space')
parser.add_argument('-r', '--random', default=123, type=int, help='random seed')

args = parser.parse_args()

# import the parameters
with open(args.params) as f:
    print('Loading the model parameters...')
    
    parameters = json.load(f)
    
    print('Model parameters loaded!')

# limit the points in the search space
print('Forming parameter grid...')

pars = list(ParameterGrid(parameters))

rs = np.random.RandomState(args.random)
plist = rs.choice(pars, args.search, replace=False).tolist()
    
with open(args.out, 'w') as f:
    json.dump(plist, f)
    
print('Parameter grid has been saved!')

