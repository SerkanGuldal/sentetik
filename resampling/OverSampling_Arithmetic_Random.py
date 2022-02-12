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

file = np.genfromtxt('datasets/yeast3_label_class.csv/' + dataname, delimiter=',', skip_header=1)
print(file.shape)
unique, counts = np.unique(file[:,-1], return_counts=True)

print('Data profile')
print(unique, counts)

minorityType_index = np.argsort(counts)[0]
majorityType_index = np.argsort(counts)[1]

minorityType = unique[minorityType_index]
majorityType = unique[majorityType_index]

minority_index = np.where(file[:,-1] == minorityType)
majority_index = np.where(file[:,-1] == majorityType)

minority = file[minority_index,:][0]
majority = file[majority_index,:][0]

N_majority = majority.shape[0]
N_minority = minority.shape[0]

print('Majority class is ' + str(majorityType) + '. Number of majority is ' + str(N_majority))
print('Minority class is ' + str(minorityType) + '. Number of minority is ' + str(N_minority))

needed =int(N_majority - N_minority)
print(str(needed) + ' datapoints need to be generated.')


synthetics = []
for i in range(needed):
    random_pair = np.random.randint(0, N_minority, 2)
    new = np.average(minority[random_pair,:], axis=0)
    new = np.ndarray.tolist(new)
    synthetics.append(new)
synthetics = np.asarray(synthetics)  
size = np.shape(synthetics)
print('Synthetically generated data size is ', size[0])

minority_increased = np.concatenate((synthetics, minority))
size = np.shape(minority_increased)
print('New total number of minority class ', size[0])

# Output file.
script_dir = os.path.abspath(os.path.dirname(sys.argv[0]) or '.')
filename = '../datasets/' + dataname + '/' + dataname + '_OverSampling_Arithmetic_Random.csv'
datasets_path = os.path.join(script_dir, filename)

new_dataset = np.concatenate((minority_increased, majority))
np.savetxt(datasets_path, new_dataset, delimiter=',')
