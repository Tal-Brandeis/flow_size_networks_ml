#!/usr/bin/env python

import xgboost
import os
import xgboost_util
import math

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import random

random.seed(0)

NUMBER_OF_TREES = 10
WINDOW_SIZE = 1
TARGET_COLUMN = 'flow_size'
TRAINING_PATH = './data/training/'
TEST_PATH = './data/test/'
VALIDATION_PATH = './data/validation/'

#paths of datasets
training_files = [os.path.join(TRAINING_PATH, f) for f in os.listdir(TRAINING_PATH)]
test_files = [os.path.join(TEST_PATH, f) for f in os.listdir(TEST_PATH)]
validation_files = [os.path.join(VALIDATION_PATH, f) for f in os.listdir(VALIDATION_PATH)]

#scaling of data to values 0-1
scaling = xgboost_util.calculate_scaling(training_files)

#preparing data for training
data_no_quantile = xgboost_util.prepare_files(training_files, WINDOW_SIZE, scaling, TARGET_COLUMN, False)
data_with_quantile = xgboost_util.prepare_files(training_files, WINDOW_SIZE, scaling, TARGET_COLUMN, True)

inputs_no_quantile, outputs_no_quantile = xgboost_util.make_io(data_no_quantile)
inputs_with_quantile, outputs_with_quantile = xgboost_util.make_io(data_with_quantile)


# fit model no training data
param = {
    'max_depth' : 5,
    'booster' : 'gbtree',
    'base_score' : 2,
    'eval_metric': 'mae'
}


#training
training_no_quantile = xgboost.DMatrix(inputs_no_quantile,outputs_no_quantile)
training_with_quantile = xgboost.DMatrix(inputs_with_quantile,outputs_with_quantile)

#build model
model_no_quantile = xgboost.train(param, training_no_quantile, 10)
model_with_quantile = xgboost.train(param, training_with_quantile, 10)


#function to print performance of iteration(train,test,validation)
def print_performance(files, quantile_active=False):
    real = []
    predicted = []
    for f in files:
        data = xgboost_util.prepare_files([f], WINDOW_SIZE, scaling, TARGET_COLUMN, quantile_active)
        #print('data\n',data)
        inputs, outputs = xgboost_util.make_io(data)
        #print('inputs\n',inputs)
        #print('outputs\n',outputs)
        if(quantile_active):
        	y_pred = model_with_quantile.predict(xgboost.DMatrix(inputs))
        	#print('y_pred_quantile_active')
        else:
        	y_pred = model_no_quantile.predict(xgboost.DMatrix(inputs))
        
        pred = y_pred.tolist()
        
        #feature_importance=model.get_score(importance_type='weight')
        #print('feature_importance\n',feature_importance)
        
        real += outputs.values.tolist()
        predicted += pred
        
    xgboost_util.print_metrics(real, predicted)


print('\nTRAINING\n')
print_performance(training_files)

print('\nTEST\n')
print_performance(test_files)

print('\nVALIDATION\n')
print_performance(validation_files)

print('\n\n-----------------------\n With quantiles\n')
print('\nTRAINING with Quantile\n')
print_performance(training_files,True)

print('\nTEST with Quantile\n')
print_performance(test_files,True)

print('\nVALIDATION with Quantile\n')
print_performance(validation_files,True)
