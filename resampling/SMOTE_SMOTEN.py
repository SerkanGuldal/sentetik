import pandas as pd
import numpy as np
from pandas import DataFrame
from collections import Counter

from imblearn.under_sampling import * 
from imblearn.over_sampling import *
from imblearn.combine import *

dataname = 'page-blocks0.csv'
NumberOfVariables = 10
data = pd.read_csv('datasets/' + dataname)
# print(data)

X = data.values[:,0:NumberOfVariables] #X tüm özellikler
y = data.values[:,NumberOfVariables] #y verinin sınıfları/label

# print("\n X Dimension is ", X.shape)
# print(X)
# print("\n y Dimension is ", y.shape)
# print(y)
print("\nOriginal class distribution is ", sorted(Counter(y).items()))

sm = eval('SMOTEN')(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
print('SMOTEN', 'resampled class distribution is ', sorted(Counter(y_res).items()))

dfx = pd.DataFrame(X_res)
dfy= pd.DataFrame(y_res)
result = pd.concat([dfx, dfy], axis=1)
# print(result)

# This output is for resampled for of the data
result.to_csv('datasets/' + dataname + '_' + 'SMOTEN' + '.csv', index=False, header=False) # First column, Index, is removed 
#result.to_csv(dataname + '_' + "SMOTEN.csv")

# This output is for information about the number of resamples
with open('datasets/' + dataname + '_' + 'SMOTEN' + '_info.txt', 'w') as filehandle:
    filehandle.write("Original distribution is " + str(sorted(Counter(y).items())))
    filehandle.write("\nResampled distribution is " + str(sorted(Counter(y_res).items())))