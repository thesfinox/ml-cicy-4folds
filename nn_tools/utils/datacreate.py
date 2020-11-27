import pandas as pd
import numpy as np

def create_features(data, rescaling=None, reshape=None):
    '''
    Create the training features (rescaled if necessary).
    
    Needed arguments:
        data: the Pandas Series with the data.
        
    Optional arguments:
        rescaling: dictionary containing the min and max rescaling parameters,
        reshape:   reshape each single feature to that shape.
    '''
    name = data.name
    
    if rescaling is not None:
        data = data.apply(lambda x: (x - rescaling['min']) / (rescaling['max'] - rescaling['min']))
    
    if reshape is not None:
        return {name: np.array([np.array(data.iloc[n]).reshape(reshape).astype(np.float32) for n in range(data.shape[0])])}
    else:
        return {name: np.array([np.array(data.iloc[n]).astype(np.float32) for n in range(data.shape[0])])}
    

def create_labels(data):
    '''
    Create the training features (rescaled if necessary).
    
    Needed arguments:
        data: the Pandas DataFrame with the data.
        
    Optional arguments:
        names: names of the labels.
    '''
        
    return {name: data[name].values.reshape(-1,).astype(np.int) for name in data.columns}