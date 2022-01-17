import numpy as np
import pandas as pd

df = pd.read_csv('../datasets/single_4_cadence_3_subsampling_3.csv')
df['AR'] = df.filename.str.split('.', expand=True).iloc[:,2]
df.drop_duplicates(subset = ['AR'], keep = 'first', inplace = True) 
print(df)

df = df.drop(columns=['AR'])
df.to_csv('../datasets/first_instance.csv', index=None)

