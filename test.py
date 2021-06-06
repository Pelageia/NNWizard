import pandas as pd
import numpy as np

data = {'first_set': [1.0,2.0,3.0,4.0,50.0,np.nan,6.0,7.0,np.nan,np.nan,8.0,9.0,10.0,np.nan],
        'second_set': ['a','b',np.nan,np.nan,'c','d','e',np.nan,np.nan,'f','g',np.nan,'h','i']
        }

df = pd.DataFrame(data,columns=['first_set','second_set'])
print(df)
mean_value=df['first_set'].mean()
df['first_set']=df['first_set'].fillna(mean_value)
print(df)