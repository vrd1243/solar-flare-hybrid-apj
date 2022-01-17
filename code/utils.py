import torch
import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import random

class Logger:
    def __init__(self, path):
        self.f = open(path, 'w')
    def log(self, log):
        print(log)
        self.f.write(log + "\n")
    def close(self):
        self.f.close()
        
def convertToString(a):
    return ' '.join(str(i) for i in a)

def create_dataframe(path, norm, feature_cols, y_col, n_harpnums=None):
    df = pd.read_csv(path, header='infer')
    df.drop_duplicates(subset="label", inplace = True)

    for column in df.columns:
        if df[column].dtype != 'float64':
            continue
        median = np.median(list(filter(lambda x: x != np.inf and x != -np.inf, df[column].values)))
        df[column] = df[column].replace([np.inf, -np.inf, np.nan], median)
    
    if norm == "minmax":
        cols_to_norm = df.columns[feature_cols]
        #cols_not_to_norm = df.columns[20:60]
        df_new = pd.DataFrame()
        df_new[cols_to_norm] = df[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        #df_new[cols_not_to_norm] = df[cols_not_to_norm]
        
    elif norm  == "standard":
        cols_to_norm = df.columns[feature_cols]
        data = df[cols_to_norm].values
        scaler = StandardScaler()
        scaler.fit(data)
        data = scaler.transform(data)
        df_new = pd.DataFrame(data, columns=[x for x in cols_to_norm])

    elif norm == "none":
        cols_to_norm = df.columns[feature_cols]
        data = df[cols_to_norm].values
        df_new = pd.DataFrame(data, columns=[x for x in cols_to_norm])

    df_new.insert(loc=0, column="label", value=df["label"].values)
    df_new['flare'] = df[y_col]

    print("NAN: ", df_new.isnull().sum().sum())
    print("INF: ", np.isinf(df_new.drop(['label'], axis=1)).values.sum())

    if n_harpnums!=None:
        pattern = re.compile('hmi.sharp_cea_720s\.(\d+)\..*')
        harpnum = df_new['label'].str.extract(pattern).astype('int64')
        harpnum_set = harpnum[0].unique()
        random.seed(1000)
        random.shuffle(harpnum_set)
        print("num of harpnums: ", harpnum_set.shape[0])
        print("num of harpnums to be selected: ", n_harpnums)
        selected_harpnums = harpnum_set[:n_harpnums]
        df_new = df_new[df_new.apply(lambda x: int(re.search(pattern, x['label']).group(1)) in selected_harpnums, axis=1)]

    return df_new

def remove_2017_plus_samples(df):

    pattern = re.compile('hmi.sharp_cea_720s\..*\.(\d\d\d\d).*')

    df_cleaned = df[df.apply(lambda x: int(re.search(pattern, x['label']).group(1)) <= 2016, axis=1)]
    return df_cleaned

    '''
    columns = ['label']
    
    for i in range(50):
        columns.append('p' + str(i))
    
    for i in range(50):
        columns.append('n' + str(i))
    
    columns += ['M_flare_in_6h', 'X_flare_in_6h', 'M_flare_in_12h', 'X_flare_in_12h','M_flare_in_24h', 'X_flare_in_24h',
                'M_flare_in_48h', 'X_flare_in_48h', 'valid']
    
    df.columns = columns
    df['any_flare_in_24h'] = df['M_flare_in_24h'] + df['X_flare_in_24h']
    rows = df.loc[:,'any_flare_in_24h'] == 2
    df.loc[rows, 'any_flare_in_24h'] = 1
    
    pattern = re.compile('hmi.sharp_cea_720s\..*\.(\d\d\d\d).*')
    df['year'] = 0
    df['year'] = df['label'].str.extract(pattern).astype('int64')
    
    
    print("These are the dataframe columns", df.columns)
    # In[87]:
    '''
    
def get_loader(set, sampler, batch_size):    
    sunspotLoader = torch.utils.data.DataLoader(set, sampler=sampler, num_workers=2, batch_size=batch_size, shuffle=True)
    return sunspotLoader
