from  utils import *
from numpy import mean
import pandas as pd
import sys
import data
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import math
import numpy as np

def ert_predict(trainX, trainY, validX, validY, impurity, feature_cols):
    
    number_of_trees = 500
    model = ExtraTreesClassifier(n_estimators = number_of_trees, random_state = 1, criterion='entropy', class_weight='balanced', min_impurity_decrease=impurity, verbose=0)
    #print(np.sum(validY[validY == 1]), np.sum(validY == 0))
    model = model.fit(trainX[:,1:], trainY)

    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_],
             axis=0)
    indices = np.argsort(importances)[::-1]

    #logger.log("Feature ranking:")
    if feature_cols is not None:
        for idx in indices:
            print(idx, feature_cols[1:][idx], importances[idx])
        #print(feature_cols[1:][indices])

    y_pred = model.predict(validX[:,1:])

    #np.save(in_dir + "/val_pred_ert_{}.npy".format(impurity), y_pred)
    #np.save(in_dir + "/val_true_ert_{}.npy".format(impurity), validY)
    #np.save(in_dir + "/val_pred_cnn_{}.npy".format(impurity), validX[:,1])
    #np.save(in_dir + "/val_filenames_{}.npy".format(impurity), validX[:,0])

    #logger.log("sum predicted : " + str(sum(y_pred)))
    #logger.log("sum actual : " + str(sum(validY)))

    #logger.log("Accuracy:" + str(metrics.accuracy_score(validY, y_pred)))

    confusion_matrix = [[0, 0], [0, 0]]

    for output, predicted in zip(validY, y_pred):

        if output == predicted and predicted == 1:
            confusion_matrix[1][1] += 1
        elif output == predicted and predicted == 0:
            confusion_matrix[0][0] += 1
        elif output != predicted and predicted == 1:
            confusion_matrix[0][1] += 1
        elif output != predicted and predicted == 0:
            confusion_matrix[1][0] += 1

    tp = confusion_matrix[1][1]
    tn = confusion_matrix[0][0]
    fp = confusion_matrix[0][1]
    fn = confusion_matrix[1][0]

    sensitivity = tp/(tp+fn)
    specificity = tn/(fp+tn)
    #precision = tp/(tp+fp)
    #recall = tp/(tp+fn)
    G_Mean = math.sqrt(sensitivity * specificity)

    #logger.log("TSS = {:.2f}".format((tp) / (tp + fn) - (fp) / (fp + tn)))
    #logger.log("Precision = {:.2f}".format((tp) / (tp + fp)))
    #logger.log("Recall = {:.2f}".format((tp) / (tp + fn)))
    #logger.log("F1 = {:.2f}".format(2*precision*recall / (precision + recall)))
    #logger.log("FB = {:.2f}".format(recall / precision))
    #logger.log("G-Mean = " + str(G_Mean))
    #logger.log("Sensitivity = " + str(sensitivity))
    #logger.log("Specificity = " + str(specificity))
    #logger.log("TP = {}, FP = {}, FN = {} TN  = {}".format(tp, fp, fn, tn))

    if tp + fp == 0:
        precision = 0
    else:
        precision = round(tp / (tp + fp),2)

    recall = round(tp / (tp + fn),2)

    tss = round(tp / (tp + fn) - fp / (fp + tn),2)

    if precision + recall == 0:
        f1 = 0
    else:    
        f1 = round(2 * precision * recall / (precision + recall),2)
    fb = round((tp + fp) / (tp + fn),2)

    return [impurity, tp, tn, fp, fn, tss, precision, recall, f1, fb]

def cnn_predict(true, pred, thresh):

    fp_idx = np.where(np.logical_and((true == 0), (pred >= thresh)))[0]
    fn_idx = np.where(np.logical_and((true == 1), (pred < thresh)))[0]
    tp_idx = np.where(np.logical_and((true == 1), (pred >= thresh)))[0]
    tn_idx = np.where(np.logical_and((true == 0), (pred < thresh)))[0]

    tp = len(tp_idx)
    fp = len(fp_idx)
    tn = len(tn_idx)
    fn = len(fn_idx)

    if tp + fp == 0:
        precision = 0
    else:
        precision = round(tp / (tp + fp),2)

    recall = round(tp / (tp + fn),2)

    tss = round(tp / (tp + fn) - fp / (fp + tn),2)

    if precision + recall == 0:
        f1 = 0
    else:    
        f1 = round(2 * precision * recall / (precision + recall),2)
    fb = round((tp + fp) / (tp + fn),2)

    return [round(thresh,3), tp, tn, fp, fn, tss, precision, recall, f1, fb]
