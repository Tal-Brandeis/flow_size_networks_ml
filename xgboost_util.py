from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
import numpy as np
from pandas import concat
import matplotlib.pyplot as plt
import os

#plot feature importance
def plot_features(features_importance_df):
	
	fig, (ax1, ax2) = plt.subplots(2, 1,figsize=(10, 10))
	ax1.bar(features_importance_df.columns,features_importance_df.values[0])
	ax2.bar(features_importance_df.columns,features_importance_df.values[3])
	
	ax1.set_title('Feature importance original method')
	ax2.set_title('Feature importance with quantile')
	
	ax1.set_ylabel('feature importance')
	#ax1.grid()
	ax2.set_ylabel('feature importance')
	#ax2.grid()
	plt.tight_layout()
	plt.savefig('plots/feature_importance.jpeg', format='jpeg' ,dpi=300)
	#plt.show()
	

#plot performance vs number of epochs
def plot_results_epochs(results_df):
	fig, (ax1, ax2, ax3) = plt.subplots(3, 1,figsize=(10, 20))
	for i in range(results_df.shape[0]):
		if(results_df.index[i].startswith('training_')):
			ax1.plot(results_df.columns,results_df.values[i], label=results_df.index[i])
		if(results_df.index[i].startswith('test_')):
			ax2.plot(results_df.columns,results_df.values[i], label=results_df.index[i])
		if(results_df.index[i].startswith('validation_')):
			ax3.plot(results_df.columns,results_df.values[i], label=results_df.index[i])
	
	ax1.set_title('Training Performance vs #Epochs')
	ax2.set_title('Test Performance vs #Epochs')
	ax3.set_title('Validation Performance vs #Epochs')
	
	ax1.set_xlabel('Number of Epochs')
	ax1.set_ylabel('$R^2$ values')
	ax1.set_xscale('log')
	ax1.legend(loc='lower right')
	ax1.grid()
	ax2.set_xlabel('Number of Epochs')
	ax2.set_ylabel('$R^2$ values')
	ax2.set_xscale('log')
	ax2.legend(loc='lower right')
	ax2.grid()
	ax3.set_xlabel('Number of Epochs')
	ax3.set_ylabel('$R^2$ values')
	ax3.set_xscale('log')
	ax3.legend(loc='lower right')
	ax3.grid()
	
	plt.tight_layout()
	plt.savefig('plots/Performance_vs_num_epochs.jpeg', format='jpeg' ,dpi=300)
	
	#plt.show()
	
#plot preformance improvment due to use of quantiles
def plot_improvement_results(results_df):
	
	fig, (ax1, ax2, ax3) = plt.subplots(3, 1,figsize=(10, 10))
	syn_columns=list(range(1,len(results_df.columns.values)+1))
	for i in range(results_df.shape[0]):
		if(results_df.index[i].startswith('training-')):
			ax1.bar(syn_columns,results_df.values[i], label=results_df.index[i],width=0.5)
		if(results_df.index[i].startswith('test-')):
			ax2.bar(syn_columns,results_df.values[i], label=results_df.index[i],width=0.5)
		if(results_df.index[i].startswith('validation-')):
			ax3.bar(syn_columns,results_df.values[i], label=results_df.index[i],width=0.5)
	
	ax1.set_title('Training Improvement due to Quantile implementation')
	ax2.set_title('Test Improvement due to Quantile implementation')
	ax3.set_title('Validation Improvement due to Quantile implementation')
	
	ax1.set_ylabel('Improvement [%]')
	ax1.set_xlabel('Number of Epochs')
	#ax1.grid()
	ax1.set_xticks(range(1,len(results_df.columns.values)+1))
	ax1.set_xticklabels(results_df.columns.values)
	
	
	ax2.set_ylabel('Improvement [%]')
	ax2.set_xlabel('Number of Epochs')
	#ax2.grid()
	ax2.set_xticks(range(1,len(results_df.columns.values)+1))
	ax2.set_xticklabels(results_df.columns.values)
	
	ax3.set_ylabel('Improvement [%]')
	ax3.set_xlabel('Number of Epochs')
	ax3.set_xticks(range(1,len(results_df.columns.values)+1))
	ax3.set_xticklabels(results_df.columns.values)
	#ax3.grid()
	
	plt.tight_layout()
	plt.savefig('plots/quantile_improvement.jpeg', format='jpeg' ,dpi=300)
	#plt.show()
	
#print results MSE, MAE, R2
def print_metrics(real, prediction):
    print('MSE: %f' % mean_squared_error(real, prediction))
    print('MAE: %f' % mean_absolute_error(real, prediction))
    print('R2: %f' % r2_score(real, prediction))
    return r2_score(real, prediction)
    
#function that returns scaling factors
def calculate_scaling(training_paths):
    scaling = {}
    #calculate scaling factors
    for f in training_paths:
        df = pd.read_csv(f, index_col=False)

        for column in df.columns:
            if column not in scaling:
               scaling[column] = 0.
            scaling[column] = max(scaling[column], float(df[column].max()))
        scaling['fs_category']=4
    return scaling

#function that resizes
def resize(s,scaling):
    return s/scaling[s.name]

#function that reads data from files and arranges it
def prepare_files(files, window_size, scaling, target_column='flow_size',quantile_active=False):
    result = []

    for f in files:
        #print('\n\n\n prepare_files\n\n\n')
        df = pd.read_csv(f, index_col=False)
        
        quantiles=df.quantile(q=[0.25, 0.5, 0.75], numeric_only=True).values.astype(float)
        
        small=float(quantiles[0][0].astype(float))
        mid=float(quantiles[1][0].astype(float))
        large=float(quantiles[2][0].astype(float))
        
        flow_size_category=np.where(df[target_column] >= large ,4, np.where(df[target_column] >= mid ,3, np.where(df[target_column] >= small ,2 ,1)))
        
        if(quantile_active):
        	df.insert(df.shape[1],'fs_category',flow_size_category)
        df = df.apply((lambda x: resize(x, scaling)), axis=0)
        
        result=df

    return result

#function that splits dataset to input and output of the ML algoritham
def make_io(data):
    #print('\n\n\n\nmake_io\n\n\n\n')
    inputs = None
    outputs = None
    
    inputs=data.iloc[:,data.columns != 'flow_size']
    outputs=data.iloc[:,data.columns == 'flow_size']
  
    return (inputs, outputs)
