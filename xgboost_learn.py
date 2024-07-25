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

NUMBER_OF_EPOCHS= 5
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
    'base_score' : 0.15,
    'eval_metric': 'mae'
}

#print('features data',data_with_quantile.columns)
#print('features inputs',inputs_with_quantile.columns)

features_inputs_with_quantile=inputs_with_quantile.columns.tolist()

#training
training_no_quantile = xgboost.DMatrix(inputs_no_quantile,outputs_no_quantile)
training_with_quantile = xgboost.DMatrix(inputs_with_quantile,outputs_with_quantile,feature_names=features_inputs_with_quantile)

#build model
model_no_quantile = xgboost.train(param, training_no_quantile, NUMBER_OF_EPOCHS)
model_with_quantile = xgboost.train(param, training_with_quantile, NUMBER_OF_EPOCHS)


#function to print performance of iteration(train,test,validation)
def print_performance(files, quantile_active=False):
    real = []
    predicted = []
    #features_to_use
    for f in files:
        data = xgboost_util.prepare_files([f], WINDOW_SIZE, scaling, TARGET_COLUMN, quantile_active)
        #print('data\n',data)
        inputs, outputs = xgboost_util.make_io(data)
        #print('inputs\n',inputs)
        #print('outputs\n',outputs)
        if(quantile_active):
        	#print('inputs',inputs)
        	y_pred = model_with_quantile.predict(xgboost.DMatrix(inputs,feature_names=features_inputs_with_quantile))
        	feature_importance=model_with_quantile.get_score(importance_type='weight')
        	print('feature_importance:',feature_importance)
        	#print('y_pred_quantile_active')
        	#print('attributes',model_with_quantile.feature_names())
        	
        else:
        	y_pred = model_no_quantile.predict(xgboost.DMatrix(inputs))
        	feature_importance=model_no_quantile.get_score(importance_type='weight')
        	print('feature_importance:',feature_importance)
        
        pred = y_pred.tolist()
        
        #feature_importance=model.get_score(importance_type='weight')
        #print('feature_importance\n',feature_importance)
        
        real += outputs.values.tolist()
        predicted += pred
        #print('real\n',real)
        #print('\npred\n',predicted)
        
    r2_temp=xgboost_util.print_metrics(real, predicted)
    return r2_temp


print('\nTRAINING\n')
r2_training_no_quantile=round(print_performance(training_files),5)

print('\nTEST\n')
r2_test_no_quantile=round(print_performance(test_files),5)

print('\nVALIDATION\n')
r2_validation_no_quantile=round(print_performance(validation_files),5)

print('\n\n-----------------------\n With quantiles\n')
print('\nTRAINING with Quantile\n')
r2_training_with_quantile=round(print_performance(training_files,True),5)

print('\nTEST with Quantile\n')
r2_test_with_quantile=round(print_performance(test_files,True),5)

print('\nVALIDATION with Quantile\n')
r2_validation_with_quantile=round(print_performance(validation_files,True),5)

print('\n\n-----------------------\n Improvment due to Quantile implementation\n')
print('TRAINING improved by', round(100*r2_training_with_quantile/r2_training_no_quantile-100,3),'%')
print('TEST improved by', round(100*r2_test_with_quantile/r2_test_no_quantile-100,3),'%')
print('VALIDATION improved by', round(100*r2_validation_with_quantile/r2_validation_no_quantile-100,3),'%')



