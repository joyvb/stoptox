#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
##########################
Modeling Notebook Tools
##########################

*Created on Wed Apr 22 16:37:35 2018 Rodolpho C. Braga*

A set of modelig tools to use in the IPython (JuPyTer) Notebook
"""

from rdkit.Chem import rdMolDescriptors
from matplotlib import cm
import numpy as np
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from array import array
import matplotlib.pyplot as plt
import argparse
import pprint
import numpy as np
import sklearn.model_selection as cv
from sklearn.utils import shuffle
from time import strftime
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer
from multiprocessing import cpu_count

from sklearn.model_selection import StratifiedKFold
import sklearn.metrics.pairwise
import scipy.spatial.distance
from io import StringIO
from sklearn.preprocessing import StandardScaler
from sklearn import pipeline, metrics
from sklearn.svm import SVC
from math import *
from sklearn.ensemble import RandomForestClassifier as RF
from rdkit.Chem import PandasTools
import sklearn.datasets
import numpy as np

from sklearn.datasets import make_regression

from sklearn.metrics import mean_squared_error
from evolutionary_search import EvolutionaryAlgorithmSearchCV
from sklearn.metrics import mean_squared_error

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import StratifiedKFold

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

os.environ['KERAS_BACKEND'] = 'tensorflow'

#from keras.models import Sequentialfrom sklearn.metrics import accuracy_score, cohen_kappa_score, matthews_corrcoef, confusion_matrix, make_scorer


import _pickle as cPickle
import pickle
from glob import glob


# Functions

def run_cv(X,y,algorithm):
    # Construct a kfolds object
    kf = StratifiedKFold(y,n_folds=5,shuffle=True)
    y_pred = y.copy()

    # Iterate through folds
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        # Initialize a classifier with key word arguments
        clf = algorithm
        clf.fit(X_train,y_train)
        y_pred[test_index] = clf.predict(X_test)
    return (y_pred, train_index, test_index)


def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(y, y_pred):
    """
    Calculates the quadratic weighted kappa
    axquadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = y
    rater_b = y_pred
    min_rating=None
    max_rating=None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return (1.0 - numerator / denominator)


def run_statistics(y,y_pred, label, coverage):

    from sklearn.metrics import confusion_matrix
    import pylab as pl
#Confusion Matrix Plot

    def draw_confusion_matrices(confusion_matrices, class_names):
        labels = list(class_names)

        for cm in confusion_matrices:
            fig = plt.figure()
            ax = fig.add_subplot(111)

            cax = ax.matshow(cm[1], cmap='PiYG')
            pl.title('Confusion Matrix\n(%s)\n' % cm[0])
            fig.colorbar(cax)
            ax.set_xticklabels([''] + labels)
            ax.set_yticklabels([''] + labels)
            pl.xlabel('Predicted Class')
            pl.ylabel('True Class')

            for i,j in ((x,y) for x in range(len(cm[1])) for y in range(len(cm[1][0]))):
                ax.annotate(str(cm[1].T[i][j]), xy=(i,j), color='black', fontweight="bold")

            pl.show()

    #from skll import Metrics

    y = np.array(y)
    class_names = np.unique(y)


    confusion_matrices = [
        ( label,  metrics.confusion_matrix(y, y_pred, ) )

    ]


    draw_confusion_matrices(confusion_matrices,class_names)
    cm=np.array(confusion_matrix(y,y_pred)).ravel()
    tn, fp, fn, tp = np.array(cm).ravel()
    print ("########################################")
    print ("#   Unbalanced Statistical Parameters  #")
    print ("########################################")
    print ("Accuracy:", round(accuracy_score(y, y_pred),2))
    print ("Area under the Curve (AUC):", round(roc_auc_score(y,y_pred),2))
    specificity = tn / (tn + fp)
    specificity = round(specificity, 2)
    sensitivity = tp / (tp + fn)
    sensitivity = round(sensitivity, 2)
    print ("########################################")
    print ("#   Balanced Statistical Parameters    #")
    print ("########################################")
    CCR= (specificity+sensitivity)/2
    CCR = round(CCR, 2)
    print ("Correct Classification Rate (CCR):", CCR)
    kappa=round(cohen_kappa_score(y,y_pred),2)
    print ("Weighted Kappa: ", kappa)
    print ("########################################")
    print ("#            Positive Class            #")
    print ("########################################")
    print ("Sensitivity (Se):", sensitivity)
    PPV =  tp / (tp + fp)
    PPV =  round(PPV, 2)
    print ("Positive Predictive Value (PPV):", PPV)
    print ("########################################")
    print ("#            Negative Class            #")
    print ("########################################")
    print ("Specificity (Sp):", specificity)
    NPV =  tn / (tn + fn)
    NPV =  round(NPV, 2)
    print ("Negative Predictive Value (NPV):", NPV)
    print ("########################################")
    print ("#     Other Statistical Parameters     #")
    print ("########################################")
    print ("Precision: ", round(precision_score(y, y_pred, average='binary'),2))
    print ("Recall: ", round(recall_score(y, y_pred, average='binary'),2))
    print ("F1: ", round(f1_score(y, y_pred, average='binary'),2))
    print ("########################################")
    print ("#        Applicability Domain          #")
    print ("########################################")
    print ("coverage: ", '{:.0f}%'.format(coverage))




def run_statistics_mult(y,y_pred, label):


#Confusion Matrix Plot

    import pylab as pl
    def draw_confusion_matrices(confusion_matrices, class_names):
        labels = list(class_names)

        for cm in confusion_matrices:
            fig = plt.figure()
            ax = fig.add_subplot(111)

            cax = ax.matshow(cm[1])
            pl.title('Confusion Matrix\n(%s)\n' % cm[0])
            fig.colorbar(cax)
            ax.set_xticklabels([''] + labels)
            ax.set_yticklabels([''] + labels)
            pl.xlabel('Predicted Class')
            pl.ylabel('True Class')

            for i,j in ((x,y) for x in range(len(cm[1])) for y in range(len(cm[1][0]))):
                ax.annotate(str(cm[1][i][j]), xy=(i,j), color='white')

            pl.show()

    #from skll import Metrics

    y = np.array(y)
    class_names = np.unique(y)


    confusion_matrices = [
        ( label,  metrics.confusion_matrix(y, y_pred) )

    ]




    # Pyplot code not included to reduce clutter
    #from churn_display import draw_confusion_matrices



    draw_confusion_matrices(confusion_matrices,class_names)
    cm=metrics.confusion_matrix(y ,y_pred)
    tp= float(cm[1,1])
    fp= float(cm[0,1])
    fn= float(cm[1,0])
    tn= float(cm[0,0])
    print ("accuracy  : ", round(accuracy_score(y, y_pred),2))
    kappa=round(cohen_kappa_score(y,y_pred),2)
    print ("Weighted Kappa   : ", kappa)
    print ("precision : ", round(precision_score(y, y_pred, average='weighted'),2))
    print ("recall    : ", round(recall_score(y, y_pred, average='weighted'),2))
    print ("f1    : ", round(f1_score(y, y_pred, average='weighted'),2))
    print ("coverage    : ", round(coverage,2))


#Applicabiltiy Domain

def getNeighborsDitance(trainingSet, testInstance, k):
    neighbors_k=sklearn.metrics.pairwise.pairwise_distances(trainingSet, Y=testInstance, metric='dice', n_jobs=1)
    neighbors_k.sort(0)
    similarity= 1-neighbors_k
    return similarity[k-1,:]


def AD_scikitlearn(x, X,y,kvar, seed, model_name='best_model'):
    seed =seed
    ypreds_folds = []
    yproba_folds = []
    yobs_folds = []
    index_train_folds = []
    index_test_folds = []
    distance_AD_train = []
    distance_AD_test = []
    training_labels_AD_final = []
    training_labels_pred_AD_final  = []
    vector_final_i_final = []
    test_index_ad_final = []
    Dcfinal = []
    k=kvar  # defided k value as in page 5 at J. Cheminform. 2013, 5, 27.
    #k_fold = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed)
    k_fold = KFold(n_splits=5, random_state=seed, shuffle=True)
    #k_fold =   StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=43)

    for train_index, test_index in k_fold.split(x,y):
        #print("TRAIN:", train_index, "TEST:", test_index)
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train, X_test = X[train_index], X[test_index]

        #print ("Training on fold " + str(train_index+1) + "/5...")
        #orignal
       #X_train, X_test = X[train_index], X[test_index]
        #

        yobs_folds.append(list(np.int32(S)[test_index]))


        model_name.fit(x_train, y_train)


        y_pred2 = np.hstack(model_name.predict(x_test))
        index_train_folds.append(list(train_index))
        index_test_folds.append(list(test_index))
        ypreds_folds.append(list(y_pred2))
        proba = np.hstack(model_name.predict_proba(x_test).max(axis=1)*100)
        yproba_folds.append(list(proba))
        #DA
        distance_AD_train_int=getNeighborsDitance(X_train, X_train, k)
        distance_AD_train.append(list(distance_AD_train_int))
        distance_AD_test_int=getNeighborsDitance(X_train, X_test, k)
        distance_AD_test.append(list(distance_AD_test_int))
        Dc=(np.average(distance_AD_train_int)-(1*(np.std(distance_AD_train_int))))
        Dcfinal.append(Dc)

        #
        vector_final_i= [i for i in range(len(test_index)) if distance_AD_test_int[i] >= Dc]
        training_labels_AD= np.int32(S)[np.array(test_index)[vector_final_i]]
        test_index_ad= (test_index)[vector_final_i] #save index test fold AD
        test_index_ad_final.append(list(test_index_ad))
        training_labels_AD_final.append(list(training_labels_AD))
        training_labels_pred_AD= y_pred2[vector_final_i]
        training_labels_pred_AD_final.append(list(training_labels_pred_AD))
        vector_final_i_final.append(list(vector_final_i))



    #fix thinfs

    ypreds_folds_total_final= np.hstack(ypreds_folds)
    index_test_folds_final = np.hstack(index_test_folds)
    training_labels_AD_final_conc = np.hstack(training_labels_AD_final)
    training_labels_pred_AD_final_conc =  np.hstack(training_labels_pred_AD_final)
    y_obs_folds=np.hstack(yobs_folds)
    test_index_ad_final_conc = np.hstack(test_index_ad_final)
    Dc = np.hstack(Dcfinal)

    print ("k-nearest neighbour distance defined to the AD      : ", k)
    print ("AD Similarity limit      : ", Dc.mean())

    return  ypreds_folds_total_final,index_test_folds_final, training_labels_AD_final_conc,training_labels_pred_AD_final_conc, y_obs_folds, test_index_ad_final_conc, Dc, index_train_folds, index_test_folds, ypreds_folds, yproba_folds, k



import pandas as pd
import pickle
from rdkit import Chem


def _tox_filter(group):
    cols = ['Prediction','Confiability']
    #cas = group['CASRN'].unique()[0]
    groupMean = group[cols].groupby('Prediction').mean()
    groupMean = groupMean.loc[groupMean.idxmax()].reset_index()
    groupMean = groupMean.iloc[0]
    groupMean.name = None

#     if groupMean.Confiability<=58:
#         result = ('Inconclusive<br> '+\
#                   '(Low)').format(**groupMean.to_dict())
#     else:

    groupMean['Outcome'] = group.Outcome.values[0]

    groupMean = groupMean[['Outcome','Prediction','Confiability']]

    #groupMean['CAS'] = cas

    groupMean = groupMean.rename(columns={'Confiability':'Confidence'})

    return groupMean

def filter_table(table):

    tableGroups = table.groupby(table.SMILES)
    tableFiltered = tableGroups.apply(_tox_filter)

    return tableFiltered
