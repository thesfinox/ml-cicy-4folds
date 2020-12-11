import pandas as pd
import numpy as np

def load_json(data, **kwargs):
    '''
    Load JSON file and output stats.
    
    Needed arguments:
        data: the JSON file (path or remote URL).
        
    Optional arguments:
        **kwargs: additional parameters to pass to pandas.read_json.
        
    Returns:
        the dataframe and the size of the dataset.
    '''
    
    df             = pd.read_json(data, **kwargs)
    n_rows, n_cols = df.shape
    
    print(f'Size of the dataset: {n_rows:d} rows, {n_cols:d} columns.')
    
    return df, (n_rows, n_cols)


def train_test(df, splits, random_state=None):
    '''
    Split the dataset into training, validation and test sets.
    
    Needed arguments:
        df:     the Pandas dataframe,
        splits: dictionary containing the fractions of the splits.
        
    Optional arguments:
        random_state: the random state.
        
    Returns:
        the splitted dataframe and the size of the folds.
        
    E.g.:
        splits = {'train_split': ..., 'val_split': ...}
        
        where `val_split` is the validation/test fraction of the remaining fold, after the training set is removed.
    '''
    
    # be agnostic on the names of the keys
    keys = list(splits.keys())
    
    # training set
    df_train = df.sample(frac=splits[keys[0]], random_state=random_state)
    n_train  = df_train.shape[0]
    
    # out of sample data
    df_oos   = df.loc[~df.index.isin(df_train.index)]

    # validation/test
    df_val   = df_oos.sample(frac=splits[keys[1]], random_state=random_state)
    n_val    = df_val.shape[0]
    
    df_test  = df_oos.loc[~df_oos.index.isin(df_val.index)]
    n_test   = df_test.shape[0]
    
    return (df_train, df_val, df_test), (n_train, n_val, n_test)
    

def remove_outliers(df, outliers=None):
    '''
    Keep values inside the outlier range.
    
    Needed arguments:
        df: the Pandas dataframe containing the data.
        
    Optional arguments:
        outliers: Pandas dataframe containing the Hodge numbers in the columns and the quantiles as index.
        
    Returns:
        the pruned dataset and its size.
        
    E.g.:
    
        outliers =
        
                 | h11 | h21 | h31 | h22 |
            ------------------------------
            low  | ... | ... | ... | ... |
            ------------------------------
            high | ... | ... | ... | ... |
            ------------------------------
    '''
    
    inliers = np.ones((df.shape[0],), dtype=bool)
    
    for label in outliers.columns:
        low_bound  = np.asarray(df[label] >= outliers[label].iloc[0], dtype=bool)
        high_bound = np.asarray(df[label] <= outliers[label].iloc[1], dtype=bool)
        inliers    = inliers & low_bound & high_bound
        
    df_new      = df.loc[inliers]
    n_train     = df.shape[0]
    n_train_new = df_new.shape[0]
    
    print(f'Samples removed: {n_train - n_train_new:d} ({100 * (n_train - n_train_new) / n_train:.2f}% of the training set)')
    
    return df, n_train_new


def create_features(data, rescaling=None, reshape=None, name=None):
    '''
    Create the features (rescaled if necessary).
    
    Needed arguments:
        data: the Pandas Series with the data.
        
    Optional arguments:
        rescaling: dictionary containing {'min': ..., 'max': ...} for rescaling,
        reshape:   tuple containing the new shape,
        name:      change name of the feature.
    '''
    if name is None:
        name = data.name
    
    if rescaling is not None:
        data = data.apply(lambda x: (x - rescaling['min']) / (rescaling['max'] - rescaling['min']))
    
    if reshape is not None:
        return {name: np.array([np.array(data.iloc[n]).reshape(reshape).astype(np.float32) for n in range(data.shape[0])])}
    else:
        return {name: np.array([np.array(data.iloc[n]).astype(np.float32) for n in range(data.shape[0])])}
    

def create_labels(data, suff=None):
    '''
    Create a dictionary of labels.
    
    Needed arguments:
        data: the Pandas DataFrame with the data,
        suff: suffix to add to the dictionary keys.
    '''
    
    if suff is not None:
        return {name + '_' + suff: data[name].values.reshape(-1,).astype(np.int) for name in data.columns}
    else:
        return {name: data[name].values.reshape(-1,).astype(np.int) for name in data.columns}
    