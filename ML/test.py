# Author Serkan Güldal 2021.09.19
from itertools import filterfalse
import os

from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

from sklearn.ensemble import AdaBoostClassifier
from imblearn.metrics import *

from collections import Counter
from sklearn.model_selection import *
from sklearn.metrics import *

from pandas import DataFrame
from pandas import read_csv
from numpy import mean
from matplotlib import pyplot

import multiprocessing as mp
import time
import xlwt
from xlwt import Workbook

import os
import sys

debug =False

wb = Workbook()
sheet1 = wb.add_sheet('Sheet 1')



def data(inputFile, NumberOfVariables): # Data importer function
    file = open(os.path.dirname(__file__) + '/../datasets/' + rawFile + '/' + inputFile)
   
    df=pd.read_csv(file)
    global X
    X = df.values[:,0:NumberOfVariables]
    global y
    y = df.values[:,NumberOfVariables]
    return(X, y)


def ml(X, y, r): # Machine learning approach
    
    if debug:
        print("Round ", r)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40)
    c = AdaBoostClassifier()
    c.fit(X_train, y_train)
    y_pred = c.predict(X_test)

    # Measurements
    Accuracy = accuracy_score(y_test, y_pred)    
    AreaUnderROCcurve = roc_auc_score(y_test, c.predict_proba(X_test)[:, 1])
    
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=0)
    AreaUndercurve0 = metrics.auc(fpr, tpr)

    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
    AreaUndercurve1 = metrics.auc(fpr, tpr)

    Recall = recall_score(y_test, y_pred, average='macro')
    Precision = precision_score(y_test, y_pred, average='macro')
    f1score = f1_score(y_test, y_pred, average='macro')
    specificity = specificity_score(y_test, y_pred, average='macro')
    sensitivity = sensitivity_score(y_test, y_pred, average=None)
    geometric = geometric_mean_score(y_test, y_pred, average=None)

    return(Accuracy, AreaUnderROCcurve, AreaUndercurve0, AreaUndercurve1, Recall, Precision, f1score, specificity, sensitivity, geometric)


def moving_average(x):
    global ave
    ave = []
    for i in range(len(x)):
        m = mean(x[0:i+1])
        ave.append(m)
    return ave


if __name__ == '__main__':
    
    print('Number of CPUs available:', mp.cpu_count())
    pool = mp.Pool()

    rawFile = 'yeast3_label_class.csv' # Filename needs to be updated!!!!
    NoV = 8 # Number of variables needs to be updated!!!!

    row = 1
    sheet1.write(0, 0, 'Method')
    sheet1.write(0, 1, 'Accuracy')
    sheet1.write(0, 2, 'Area Under ROC curve')
    sheet1.write(0, 3, 'Area Under the Curve 0')
    sheet1.write(0, 4, 'Area Under the Curve 1')
    sheet1.write(0, 5, 'Recall')
    sheet1.write(0, 6, 'Precision')
    sheet1.write(0, 7, 'F1 Score')
    sheet1.write(0, 8, 'Specificity')
    sheet1.write(0, 9, 'Sensivity')
    sheet1.write(0, 10, 'Geometric Mean')
    sheet1.write(0, 11, 'Arithmetic Mean')
    sheet1.write(0, 12, 'Total time')

    for file in [
        '',
        # '_ADASYN.csv',
        # '_AllKNN.csv',
        # '_BorderlineSMOTE.csv',
        # '_ClusterCentroids.csv',
        # '_CondensedNearestNeighbour.csv',
        # '_EditedNearestNeighbours.csv',
        # '_InstanceHardnessThreshold.csv',
        # '_NearMiss.csv',
        # '_NeighbourhoodCleaningRule.csv',
        # '_OneSidedSelection.csv',
        # '_RandomOverSampler.csv',
        # '_RandomUnderSampler.csv',
        # '_RepeatedEditedNearestNeighbours.csv',
        # '_SMOTE.csv',
        # '_SMOTEENN.csv',
        # '_SMOTEN.csv',
        # '_SMOTETomek.csv',
        # '_SVMSMOTE.csv',
        # '_TomekLinks.csv',
        # #Our methods
        # '_AGG_WA_CD.csv',
        # '_GM_WA_CD.csv',
        # '_Heinz.csv',
        '_Weighted.csv']:

        if file == '':
            print('Raw')
        else:
            print(file[1:-4])

        data(rawFile + file, NoV)

        ts = time.time()

        for r in range(10):
            print(ml(X, y, r))