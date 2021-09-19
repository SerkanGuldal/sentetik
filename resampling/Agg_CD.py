# Author Serkan Güldal 2021.09.19
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import collections

import os
import sys

dataname = 'yeast3_label_class.csv' # Filename needs to be updated!!!!
df = pd.read_csv('datasets/yeast3/' + dataname)
distribution = collections.Counter(df.iloc[:,-1]) #Label needs to be selected 
print('Data profile')
print(distribution)

majority = distribution.most_common()[0][0]
minority = distribution.most_common()[1][0]

N_majority = distribution.most_common()[0][1]
N_minority = distribution.most_common()[1][1]
N_middle = N_minority + (N_majority - N_minority)/2

print('Majority class is ' + str(majority) + '. Number of majority is ' + str(N_majority))
print('Minority class is ' + str(minority) + '. Number of minority is ' + str(N_minority))
print('Middle point is ' + str(N_middle))

print(str(N_majority - N_middle) + ' needs to be reduced.')
print(str(N_middle - N_minority) + ' needs to be generated.')
df_majority = df.loc[df[' Class']== majority]

model = AgglomerativeClustering(n_clusters=6, affinity='euclidean', linkage='ward')
model.fit(df_majority)
labels = model.labels_
clustersPop = collections.Counter(labels)  # Cluster populations
print('Clusters are arranged as:')
print(clustersPop)

print(df_majority[labels == 2].shape)
df_majority_clustered = df_majority[labels == 2]
df_majority_clustered.to_csv(dataname + '_AGG.csv')

script_dir = os.path.abspath(os.path.dirname(sys.argv[0]) or '.')
filename = '../datasets/' + dataname + '_AGG.csv'
datasets_path = os.path.join(script_dir, filename)
df_majority_clustered.to_csv(datasets_path)