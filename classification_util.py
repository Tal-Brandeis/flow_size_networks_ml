from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
import numpy as np
from pandas import concat
import matplotlib.pyplot as plt
import os

#plot false classify helper, count false classify cases
def check_false_classify(real,pred):
	real_arr=np.array(real)
	pred_arr=np.array(pred)
	temp=np.zeros(len(real))
	
	for i in range(len(real)):
		if(pred_arr[i]>real_arr[i]):
			temp[i]=3
		if(pred_arr[i]==real_arr[i]):
			temp[i]=2
		if(pred_arr[i]<real_arr[i]):
			temp[i]=1	
				
	
	
	greater=(temp==3).sum()
	exact=(temp==2).sum()
	smaller=(temp==1).sum()
	
	return greater,exact,smaller
	
#plot false classify	
def plot_false_classify(fc_df):
	
	labels='Greater','Exact', 'Smaller'
	colors=['gray','cyan','rosybrown']
	
	
	fig, (ax1, ax2, ax3) = plt.subplots(3, 1,figsize=(10, 20))
	for i in range(fc_df.shape[0]):
		if(fc_df.index[i].startswith('training')):
			ax1.pie(fc_df.values[i],labels=labels,colors=colors, autopct='%1.1f%%')
		if(fc_df.index[i].startswith('test')):
			ax2.pie(fc_df.values[i],labels=labels,colors=colors, autopct='%1.1f%%')
		if(fc_df.index[i].startswith('validation')):
			ax3.pie(fc_df.values[i],labels=labels,colors=colors, autopct='%1.1f%%')
	
	ax1.set_title('Training classify')
	ax2.set_title('Test classify')
	ax3.set_title('Validation classify')
	#plt.show()
	plt.tight_layout()
	
	plt.savefig('plots/classification_false_classify.jpeg', format='jpeg' ,dpi=300)
	
		
#plot feature importance
def plot_features(features_importance_df):
	
	plt.figure()
	plt.bar(features_importance_df.columns,features_importance_df.values[0])
	plt.title('Feature importance classification')
	plt.ylabel('feature importance')

	plt.savefig('plots/classifiction_feature_importance.jpeg', format='jpeg' ,dpi=300)
	#plt.show()
	

#plot performance vs number of epochs
def plot_results_epochs(results_df):
	fig, (ax1, ax2, ax3) = plt.subplots(3, 1,figsize=(10, 20))
	for i in range(results_df.shape[0]):
		if(results_df.index[i].startswith('training')):
			ax1.plot(results_df.columns,results_df.values[i], label=results_df.index[i])
		if(results_df.index[i].startswith('test')):
			ax2.plot(results_df.columns,results_df.values[i], label=results_df.index[i])
		if(results_df.index[i].startswith('validation')):
			ax3.plot(results_df.columns,results_df.values[i], label=results_df.index[i])
	
	ax1.set_title('Training Performance vs #Epochs')
	ax2.set_title('Test Performance vs #Epochs')
	ax3.set_title('Validation Performance vs #Epochs')
	
	ax1.set_xlabel('Number of Estimators')
	ax1.set_ylabel('$R^2$ values')
	ax1.set_xscale('log')
	ax1.legend(loc='lower right')
	ax1.grid()
	ax2.set_xlabel('Number of Estimators')
	ax2.set_ylabel('$R^2$ values')
	ax2.set_xscale('log')
	ax2.legend(loc='lower right')
	ax2.grid()
	ax3.set_xlabel('Number of Estimators')
	ax3.set_ylabel('$R^2$ values')
	ax3.set_xscale('log')
	ax3.legend(loc='lower right')
	ax3.grid()
	
	plt.tight_layout()
	plt.savefig('plots/classification_Performance_vs_num_epochs.jpeg', format='jpeg' ,dpi=300)
	
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
            #scaling[column] = 1
        #scaling['fs_category']=3
        scaling['fs_category']=1
    return scaling

#function that resizes
def resize(s,scaling):
    return s/scaling[s.name]

#function that reads data from files and arranges it
def prepare_files(files, scaling, target_column='fs_category'):
    result = []

    for f in files:
        #print('\n\n\n prepare_files\n\n\n')
        df = pd.read_csv(f, index_col=False)

        
        quantiles=df.quantile(q=[0.25, 0.5, 0.75], numeric_only=True).values.astype(float)
        
        small=float(quantiles[0][0].astype(float))
        mid=float(quantiles[1][0].astype(float))
        large=float(quantiles[2][0].astype(float))
        
        flow_size_category=np.where(df['flow_size'] >= large ,3, np.where(df['flow_size'] >= mid ,2, np.where(df['flow_size'] >= small ,1 ,0)))
        
        df.insert(df.shape[1],'fs_category',flow_size_category)
        df = df.apply((lambda x: resize(x, scaling)), axis=0)
        

        result=df

    return result

#function that splits dataset to input and output of the ML algoritham
def make_io(data):
    #print('\n\n\n\nmake_io\n\n\n\n')
    inputs = None
    outputs = None

    inputs=data.iloc[:,data.columns != 'fs_category']
    inputs=inputs.iloc[:,inputs.columns != 'flow_size']
    
    outputs=data.iloc[:,data.columns == 'fs_category']
   
    return (inputs, outputs)
