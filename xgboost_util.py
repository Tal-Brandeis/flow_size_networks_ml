from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
import numpy as np
from pandas import concat
FLOW_SIZE_THREASH = 1000

def print_metrics(real, prediction):
    print('MSE: %f' % mean_squared_error(real, prediction))
    print('MAE: %f' % mean_absolute_error(real, prediction))
    print('R2: %f' % r2_score(real, prediction))

def calculate_scaling(training_paths):
    scaling = {}
    #calculate scaling factors
    for f in training_paths:
        df = pd.read_csv(f, index_col=False)

        for column in df.columns:
            if column not in scaling:
               scaling[column] = 0.
            scaling[column] = max(scaling[column], float(df[column].max()))
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
        small=float(quantiles[0].astype(float))
        mid=float(quantiles[1].astype(float))
        large=float(quantiles[2].astype(float))
        #print('small',small)
        #print('mid',mid)
        #print('large',large)
        #print('datat_type',type(a))
        #print('describe\n',df.describe(numeric_only=True).loc['25%'])
        
        #flow_size_category=np.where(df[target_column] > FLOW_SIZE_THREASH, 1,-1)
        #flow_size_category=np.where(df[target_column] >= float(quantiles[2]),2,1)
        flow_size_category=np.where(df[target_column] >= large ,4, np.where(df[target_column] >= mid ,3, np.where(df[target_column] >= small ,2 ,1)))
        
        #print('flow_size_category\n',flow_size_category)
        #print('flow_size_category len\n',len(flow_size_category))
        #print('4 quantile 0.75-1',(flow_size_category==4).sum())
        #print('3 quantile 0.5-0.75',(flow_size_category==3).sum())
        #print('2 quantile 0.25-0.5',(flow_size_category==2).sum())
        #print('1 quantile 0-0.25',(flow_size_category==1).sum())

        df = df.apply((lambda x: resize(x, scaling)), axis=0)
        #print('df_after scale\n',df)

        result=df
        if(quantile_active):
        	result.insert(1,'flow_size_category',flow_size_category)
        
        #print('result',result)

    return result

def make_io(data):
    #print('\n\n\n\nmake_io\n\n\n\n')
    inputs = None
    outputs = None
    #print('data\n',data)

    #inputs=data.iloc[:,0]
    inputs=data
    
    #print('inputs\n',inputs)
    outputs=data
    #outputs=data.iloc[:,0]
    #outputs=data.iloc[:,-1]
    #print('\noutputs\n',outputs)
    #print('\noutputs[1]\n',data.iloc[1])
    #print('outputs type\n',type(outputs))
    
    #outputs=data.values.tolist()
    #print('\noutputs\n',outputs)    
    return (inputs, outputs)
