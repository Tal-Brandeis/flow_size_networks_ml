from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
import numpy as np
from pandas import concat
import matplotlib.pyplot as plt
import os

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
	


def print_metrics(real, prediction):
    print('MSE: %f' % mean_squared_error(real, prediction))
    print('MAE: %f' % mean_absolute_error(real, prediction))
    print('R2: %f' % r2_score(real, prediction))
    return r2_score(real, prediction)

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

def resize(s,scaling):
    return s/scaling[s.name]

def prepare_files(files, window_size, scaling, target_column='flow_size',quantile_active=False):
    result = []

    for f in files:
        #print('\n\n\n prepare_files\n\n\n')
        df = pd.read_csv(f, index_col=False)
        #print('df.read',df)

        #df = df.drop("index", axis=1)
        #print('quantile\n',df.quantile(q=[0.25, 0.5, 0.75], numeric_only=True).values)
        quantiles=df.quantile(q=[0.25, 0.5, 0.75], numeric_only=True).values.astype(float)
        #print('quantiles',quantiles)
        small=float(quantiles[0][0].astype(float))
        mid=float(quantiles[1][0].astype(float))
        large=float(quantiles[2][0].astype(float))
        #print('small',small)
        #print('mid',mid)
        #print('large',large)
        flow_size_category=np.where(df[target_column] >= large ,4, np.where(df[target_column] >= mid ,3, np.where(df[target_column] >= small ,2 ,1)))
        
        #print('flow_size_category\n',flow_size_category)
        #print('flow_size_category len\n',len(flow_size_category))
        #print('4 quantile 0.75-1',(flow_size_category==4).sum())
        #print('3 quantile 0.5-0.75',(flow_size_category==3).sum())
        #print('2 quantile 0.25-0.5',(flow_size_category==2).sum())
        #print('1 quantile 0-0.25',(flow_size_category==1).sum())
        if(quantile_active):
        	df.insert(df.shape[1],'fs_category',flow_size_category)
        df = df.apply((lambda x: resize(x, scaling)), axis=0)
        #print('df_after scale\n',df)
        

        result=df
        #print('result.shape',result.shape[1])
        #if(quantile_active):
        #	result.insert(result.shape[1],'fs_category',flow_size_category)
        
        #print('result',result)

    return result

def make_io(data):
    #print('\n\n\n\nmake_io\n\n\n\n')
    inputs = None
    outputs = None
    #print('data\n',data.iloc[:,data.columns != 'flow_size'])
    #print('data type',type(data))

    inputs=data.iloc[:,data.columns != 'flow_size']
    #inputs=data
    
    #print('inputs\n',inputs)
    #outputs=data
    #outputs=data.iloc[:,0]
    #outputs=data.iloc[:,-1]
    outputs=data.iloc[:,data.columns == 'flow_size']
    #print('\noutputs\n',outputs)
    #print('\noutputs[1]\n',data.iloc[1])
    #print('outputs type\n',type(outputs))
    
    #outputs=data.values.tolist()
    #print('\noutputs\n',outputs)    
    return (inputs, outputs)
