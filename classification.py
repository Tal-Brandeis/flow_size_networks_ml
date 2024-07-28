#!/usr/bin/env python

import xgboost
import os
import classification_util
import math
import time

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import random
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

random.seed(0)

NUMBER_OF_EPOCHS= [1,2,3,5,10,100,1000]
#NUMBER_OF_EPOCHS= [10]

TESTS_RUN=['training','test','validation']

TARGET_COLUMN = 'fs_category'
TRAINING_PATH = './data/training/'
TEST_PATH = './data/test/'
VALIDATION_PATH = './data/validation/'

#paths of datasets
training_files = [os.path.join(TRAINING_PATH, f) for f in os.listdir(TRAINING_PATH)]
test_files = [os.path.join(TEST_PATH, f) for f in os.listdir(TEST_PATH)]
validation_files = [os.path.join(VALIDATION_PATH, f) for f in os.listdir(VALIDATION_PATH)]

#scaling of data to values 0-1
scaling = classification_util.calculate_scaling(training_files)

#preparing data for training
data = classification_util.prepare_files(training_files, scaling, TARGET_COLUMN)

inputs, outputs = classification_util.make_io(data)


# fit model no training data
param = {
    'max_depth' : 5,
    'booster' : 'gbtree',
    'base_score' : 0.5,
    'eval_metric': 'mae'
}



#features used in model
features=inputs.columns.tolist()
features_importance_arr=np.zeros((len(TESTS_RUN),len(features)))
features_importance_df=pd.DataFrame(features_importance_arr, columns=features, index=TESTS_RUN)

results_arr=np.zeros((len(TESTS_RUN),len(NUMBER_OF_EPOCHS)))
results_df=pd.DataFrame(results_arr, columns=NUMBER_OF_EPOCHS, index=TESTS_RUN)

fc_columns=['greater','exact','smaller']
false_classify_arr=np.zeros((len(TESTS_RUN),3))
false_classify_df=pd.DataFrame(false_classify_arr, columns=fc_columns, index=TESTS_RUN)
#print('false_classify_df\n',false_classify_df)



#main
def main(num_epochs):
	#training
	training = xgboost.DMatrix(inputs,outputs)

	#build model
	model= xgboost.train(param, training, num_epochs)

	
	#function to print performance of iteration(train,test,validation)
	def print_performance(files):
	    real = []
	    predicted = []
	    
	    #features_to_use
	    for f in files:
	    	data = classification_util.prepare_files([f], scaling, TARGET_COLUMN)
	    	inputs, outputs = classification_util.make_io(data)
	    	y_pred = model.predict(xgboost.DMatrix(inputs))
	    	feature_importance=model.get_score(importance_type='weight')
		
	    	pred = y_pred.tolist()
	    	
		
	    	real += outputs.values.tolist()
	    	predicted += pred
	    	
	    r2_temp=classification_util.print_metrics(real, predicted)
	    #print('real\n',real)
	    #print('predicted\n',predicted)
	    #print(len(predicted))
	    #print(len(real))
	    temp_multi=4*np.ones(len(predicted))
	    #print('temp_multi',temp_multi)
	    temp_predicted=temp_multi*predicted
	    #print('temp_predicted\n',temp_predicted)
	    temp_predicted=np.round(temp_predicted)
	    #print('temp_predicted\n',temp_predicted)
	    temp_predicted=temp_predicted/temp_multi
	    #sprint('temp_predicted\n',temp_predicted)
	    #print('real',real)
	    g,e,s=classification_util.check_false_classifiy(real,temp_predicted)
	    #if(i==10):
	    #	classification_util.plot_false_classifiy(g,e,s)
	    return r2_temp,feature_importance,g,e,s
	

	print('\nTRAINING\n')
	r2_training, fi_training,g_training,e_training,s_training=print_performance(training_files)
	r2_training=round(r2_training,5)
	results_df[i]['training']=r2_training
	
	
	print('\nTEST\n')
	r2_test,fi_test,g_test,e_test,s_test=print_performance(test_files)
	r2_test=round(r2_test,5)
	results_df[i]['test']=r2_test

	print('\nVALIDATION\n')
	r2_validation,fi_validation,g_validation,e_validation,s_validation=print_performance(validation_files)
	r2_validation=round(r2_validation,5)
	results_df[i]['validation']=r2_validation

	
	if(i==10):
		features_importance_df.loc['training']=fi_training
		features_importance_df.loc['test']=fi_test
		features_importance_df.loc['validation']=fi_validation
		
		false_classify_df.loc['training']=[g_training,e_training,s_training]
		false_classify_df.loc['test']=[g_test,e_test,s_test]
		false_classify_df.loc['validation']=[g_validation,e_validation,s_validation]
		#print('false_classify_df\n',false_classify_df)
		classification_util.plot_false_classifiy(false_classify_df)
		
		
		

	


#run iterations on NUMBER_OF_EPOCHS

for i in NUMBER_OF_EPOCHS:
	print('\n\n\n------------NUMBER_OF_EPOCHS:',i,'-----------')
	main(i)


#print and save results
#print('\nresults_df\n',results_df)
results_df.to_csv('classification_results_performance_epochs.csv')
classification_util.plot_results_epochs(results_df)

#print('features_importance_df\n',features_importance_df.fillna(0))
classification_util.plot_features(features_importance_df.fillna(0))





	


