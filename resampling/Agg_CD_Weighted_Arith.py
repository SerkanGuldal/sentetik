# Author Serkan Güldal 2021.09.19
from sklearn.cluster import AgglomerativeClustering
import scipy
import pandas as pd
import collections
import numpy as np
import os
import sys
import random

dataname = 'yeast3_label_class.csv' # Filename needs to be updated!!!!
df = pd.read_csv('datasets/yeast3/' + dataname)
distribution = collections.Counter(df.iloc[:,-1]) #Label needs to be selected 
print('Data profile')
print(distribution)

majorityType = distribution.most_common()[0][0]
minorityType = distribution.most_common()[1][0]

N_majority = distribution.most_common()[0][1]
N_minority = distribution.most_common()[1][1]
N_middle = N_minority + (N_majority - N_minority)/2

print('Majority class is ' + str(majorityType) + '. Number of majority is ' + str(N_majority))
print('Minority class is ' + str(minorityType) + '. Number of minority is ' + str(N_minority))
print('Middle point is ' + str(N_middle))


excess = int(N_majority - N_middle)
print(str(excess) + ' needs to be reduced.')

needed =int(N_middle - N_minority)
print(str(needed) + ' needs to be generated.')

# Agglomerate to reduce
df_majority = df.loc[df[' Class']== majorityType]
model = AgglomerativeClustering(n_clusters=6, affinity='euclidean', linkage='ward')
model.fit(df_majority)
labels = model.labels_
clustersPop = collections.Counter(labels)  # Cluster populations
print('Clusters are arranged as:')
print(clustersPop)

df_majority_clustered = df_majority[labels == 2]

#Correlation distance to increase
df_minority = df.loc[df[' Class'] == minorityType]
pair = []
for i in range(len(df_minority)):
    for j in range(i,len(df_minority)):
        if i != j:
            cd = [i, j, scipy.spatial.distance.correlation(df_minority.iloc[i],df_minority.iloc[j])]
            pair.append(cd)

details = pd.DataFrame(pair, columns=['pair1', 'pair2', 'Distance'])
pairSorted = details.sort_values(by = 'Distance')
pairCoor = pairSorted.iloc[:needed]
# print(pairCoor)
# size = np.shape(pairCoor)
# print(size)

synthetics = []
for i in range(len(pairCoor)):
    synthetic = []
    for j in range(len(df.columns)):
        pair1 = df_minority.iloc[pairCoor.iloc[i,0],j]
        pair2 = df_minority.iloc[pairCoor.iloc[i,1],j]
        vector = ([pair1, pair2])
        weight = random.random()
        weighted = np.average(vector, axis=0, weights=([weight, 1-weight]))
        synthetic.append(weighted)
    synthetics.append(synthetic)

size = np.shape(synthetics)
print(size)

df_minority_increased = np.concatenate((synthetics, df_minority))
size = np.shape(df_minority_increased)
print(size)

# Output file.
script_dir = os.path.abspath(os.path.dirname(sys.argv[0]) or '.')
filename = '../datasets/' + dataname + '_AGG.csv'
datasets_path = os.path.join(script_dir, filename)

df_new = np.concatenate((df_minority_increased, df_majority_clustered))

pd.DataFrame(df_new).to_csv(datasets_path, header=False)