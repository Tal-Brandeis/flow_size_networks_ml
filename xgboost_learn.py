#!/usr/bin/env python

import xgboost
import os
import xgboost_util
import math

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import random
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

random.seed(0)

NUMBER_OF_EPOCHS= [1,2,3,5,10,100,1000]
#TESTS_RUN=['training_no_quantile','test_no_quantile','validation_no_quantile','training_with_quantile','test_with_quantile','validation_with_quantile']
TESTS_RUN=['training_no_quantile','test_no_quantile','validation_no_quantile','training_with_quantile','test_with_quantile','validation_with_quantile','training-improvement','test-improvement','validation-improvement']

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



#features used in model
features_inputs_with_quantile=inputs_with_quantile.columns.tolist()

results_arr=np.zeros((len(TESTS_RUN),len(NUMBER_OF_EPOCHS)))
results_df=pd.DataFrame(results_arr, columns=NUMBER_OF_EPOCHS, index=TESTS_RUN)
#results_df[10]['training_no_quantile']=3
#print('results_df',results_df)

#main
def main(num_epochs):
	#training
	training_no_quantile = xgboost.DMatrix(inputs_no_quantile,outputs_no_quantile)
	training_with_quantile = xgboost.DMatrix(inputs_with_quantile,outputs_with_quantile,feature_names=features_inputs_with_quantile)


	#build model
	model_no_quantile = xgboost.train(param, training_no_quantile, num_epochs)
	model_with_quantile = xgboost.train(param, training_with_quantile, num_epochs)


	#function to print performance of iteration(train,test,validation)
	def print_performance(files, quantile_active=False):
	    real = []
	    predicted = []
	    #features_to_use
	    for f in files:
	    	data = xgboost_util.prepare_files([f], WINDOW_SIZE, scaling, TARGET_COLUMN, quantile_active)
	    	inputs, outputs = xgboost_util.make_io(data)
	    	if(quantile_active):
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
	results_df[i]['training_no_quantile']=r2_training_no_quantile
	
	print('\nTEST\n')
	r2_test_no_quantile=round(print_performance(test_files),5)
	results_df[i]['test_no_quantile']=r2_test_no_quantile

	print('\nVALIDATION\n')
	r2_validation_no_quantile=round(print_performance(validation_files),5)
	results_df[i]['validation_no_quantile']=r2_validation_no_quantile

	print('\n\n-----------------------\n With quantiles\n')
	print('\nTRAINING with Quantile\n')
	r2_training_with_quantile=round(print_performance(training_files,True),5)
	results_df[i]['training_with_quantile']=r2_training_with_quantile

	print('\nTEST with Quantile\n')
	r2_test_with_quantile=round(print_performance(test_files,True),5)
	results_df[i]['test_with_quantile']=r2_test_with_quantile

	print('\nVALIDATION with Quantile\n')
	r2_validation_with_quantile=round(print_performance(validation_files,True),5)
	results_df[i]['validation_with_quantile']=r2_validation_with_quantile

	print('\n\n-----------------------\n Improvment due to Quantile implementation\n')
	results_df[i]['training-improvement']=round(100*r2_training_with_quantile/r2_training_no_quantile-100,3)
	results_df[i]['test-improvement']=round(100*r2_test_with_quantile/r2_test_no_quantile-100,3)
	results_df[i]['validation-improvement']=round(100*r2_validation_with_quantile/r2_validation_no_quantile-100,3)
	
	print('TRAINING improved by', round(100*r2_training_with_quantile/r2_training_no_quantile-100,3),'%')
	print('TEST improved by', round(100*r2_test_with_quantile/r2_test_no_quantile-100,3),'%')
	print('VALIDATION improved by', round(100*r2_validation_with_quantile/r2_validation_no_quantile-100,3),'%')
	


#run iterations on NUMBER_OF_EPOCHS
for i in NUMBER_OF_EPOCHS:
	print('\n\n\n------------NUMBER_OF_EPOCHS:',i,'-----------')
	main(i)


#print and save results
print('results_df\n',results_df)
results_df.to_csv('results_performance_epochs.csv')
xgboost_util.plot_results_epochs(results_df)


	


