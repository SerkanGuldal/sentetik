import pandas as pd
import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTENC

dataname = 'page-blocks0.csv'
NumberOfVariables = 10
data = pd.read_csv('datasets/' + dataname)
# print(data)

X = data.values[:, 0:NumberOfVariables]  # X contains all features
y = data.values[:, NumberOfVariables]  # y contains class labels

# print("\n X Dimension is ", X.shape)
# print(X)
# print("\n y Dimension is ", y.shape)
# print(y)
print("\nOriginal class distribution is ", sorted(Counter(y).items()))

categorical_features_indices = [-1]  # Define indices of categorical features if any

sm = SMOTENC(categorical_features=categorical_features_indices, random_state=42)
X_res, y_res = sm.fit_resample(X, y)
print('SMOTENC', 'resampled class distribution is ', sorted(Counter(y_res).items()))

dfx = pd.DataFrame(X_res)
dfy = pd.DataFrame(y_res)
result = pd.concat([dfx, dfy], axis=1)
# print(result)

# Output the resampled data
result.to_csv('datasets/' + dataname + '_' + 'SMOTENC' + '.csv', index=False, header=False)  # Remove the first column (index)

# Output information about the number of resamples
with open('datasets/' + dataname + '_' + 'SMOTENC' + '_info.txt', 'w') as filehandle:
    filehandle.write(" Original distribution is " + str(sorted(Counter(y).items())))
    filehandle.write("\nResampled distribution is " + str(sorted(Counter(y_res).items())))