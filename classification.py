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
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
warnings.simplefilter(action='ignore', category=FutureWarning)

random.seed(0)

NUMBER_OF_ESTIMATORS= [5,10,100,1000]
#NUMBER_OF_ESTIMATORS= [100]

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



#features used in model
features=inputs.columns.tolist()
features_importance_arr=np.zeros((len(TESTS_RUN),len(features)))
features_importance_df=pd.DataFrame(features_importance_arr, columns=features, index=TESTS_RUN)

results_arr=np.zeros((len(TESTS_RUN),len(NUMBER_OF_ESTIMATORS)))
results_df=pd.DataFrame(results_arr, columns=NUMBER_OF_ESTIMATORS, index=TESTS_RUN)

fc_columns=['greater','exact','smaller']
false_classify_arr=np.zeros((len(TESTS_RUN),3))
false_classify_df=pd.DataFrame(false_classify_arr, columns=fc_columns, index=TESTS_RUN)
#print('false_classify_df\n',false_classify_df)



#main
def main(num):

	#labels
	le=LabelEncoder()
	y=le.fit_transform(np.ravel(outputs))
	#model
	model=RandomForestClassifier(n_estimators=num, random_state=42)
	#fit
	model.fit(inputs,y)
	
	#function to print performance of iteration(train,test,validation)
	def print_performance(files):
	    real = []
	    predicted = []
	    
	    #features_to_use
	    for f in files:
	    	data = classification_util.prepare_files([f], scaling, TARGET_COLUMN)
	    	inputs, outputs = classification_util.make_io(data)
	    	y_pred = model.predict(inputs)
	    	pred = y_pred.tolist()
	    	
	    	real += outputs.values.tolist()
	    	predicted += pred
		
	    #print results MSE, MAE, R2
	    r2_temp=classification_util.print_metrics(real, predicted)
	    
	    g,e,s=classification_util.check_false_classify(real,predicted)
	    feature_importance=[]
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

	#plot false classify
	if(i==100):
		
		false_classify_df.loc['training']=[g_training,e_training,s_training]
		false_classify_df.loc['test']=[g_test,e_test,s_test]
		false_classify_df.loc['validation']=[g_validation,e_validation,s_validation]
		classification_util.plot_false_classify(false_classify_df)
		
		
		

	


#run iterations on NUMBER_OF_ESTIMATORS

for i in NUMBER_OF_ESTIMATORS:
	print('\n\n\n------------NUMBER_OF_ESTIMATORS:',i,'-----------')
	main(i)


#print and save results
#print('\nresults_df\n',results_df)
#save results to csv
results_df.to_csv('results/classification_results_performance_epochs.csv')
#plot results
classification_util.plot_results_epochs(results_df)



