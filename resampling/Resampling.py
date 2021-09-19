import pandas as pd
import numpy as np
from pandas import DataFrame
from collections import Counter

from imblearn.under_sampling import * 
from imblearn.over_sampling import *
from imblearn.combine import *

dataname = 'yeast3_label_class.csv'
NumberOfVariables = 8
data = pd.read_csv(dataname)
print(data)

X = data.values[:,0:NumberOfVariables] #X tüm özellikler
y = data.values[:,NumberOfVariables] #y verinin sınıfları/label

print("\n X Dimension is ", X.shape)
print(X)
print("\n y Dimension is ", y.shape)
print(y)
print("\nOriginal class distribution is ", sorted(Counter(y).items()))

for method in [
    'ADASYN',
    'BorderlineSMOTE',
    'ClusterCentroids',
    'CondensedNearestNeighbour',
    'RandomOverSampler',
    'RandomUnderSampler',
    'SMOTE',
    'SMOTEENN',
    'SMOTEN',
    'SMOTETomek',
    'SVMSMOTE']:
            
    sm = eval(method)(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    print(method, 'resampled class distribution is ', sorted(Counter(y_res).items()))

    dfx = pd.DataFrame(X_res)
    dfy= pd.DataFrame(y_res)
    result = pd.concat([dfx, dfy], axis=1)
    # print(result)

    result.to_csv(dataname + '_' + method + '.csv', index=False) # First column, Index, is removed 
    #result.to_csv(dataname + '_' + "SMOTE.csv")

    with open(dataname + '_' + method + '_info.csv', 'w') as filehandle:
        filehandle.write(" Original distribution is " + str(sorted(Counter(y).items())))
        filehandle.write("\nResampled distribution is " + str(sorted(Counter(y_res).items())))

for method in [
    'AllKNN',
    'EditedNearestNeighbours',
    'InstanceHardnessThreshold',
    'NearMiss',
    'NeighbourhoodCleaningRule',
    'OneSidedSelection',
    'RepeatedEditedNearestNeighbours',
    'TomekLinks']:
            
    sm = eval(method)()
    X_res, y_res = sm.fit_resample(X, y)
    print(method, 'resampled class distribution is ', sorted(Counter(y_res).items()))

    dfx = pd.DataFrame(X_res)
    dfy= pd.DataFrame(y_res)
    result = pd.concat([dfx, dfy], axis=1)
    # print(result)

    result.to_csv(dataname + '_' + method + '.csv', index=False) # First column, Index, is removed 
    #result.to_csv(dataname + '_' + "SMOTE.csv")

    with open(dataname + '_' + method + '_info.csv', 'w') as filehandle:
        filehandle.write(" Original distribution is " + str(sorted(Counter(y).items())))
        filehandle.write("\nResampled distribution is " + str(sorted(Counter(y_res).items())))