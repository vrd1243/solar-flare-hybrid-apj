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
import time
import math
import numpy as np
from predict import ert_predict, cnn_predict

in_dir = sys.argv[1]

which_features = sys.argv[2] 

outputPath = in_dir

logger = Logger(outputPath + "/dnn_cubical_complexes_decision_trees_" + which_features + ".out")
    
logger.log("which features=" + which_features)

df = pd.read_csv(in_dir + '/train.csv')

#top_cadence = 40
top_cadence = 60
if which_features == "topological":
    n_components = 2
    cols = list(range(0,2)) + list(range(27+251,27+502,top_cadence)) + list(range(27+501+251, df.shape[1],top_cadence))#df.shape[1]))
    pca_cols = list(range(27,df.shape[1]))#df.shape[1]))

elif which_features == "traditional":
    n_components = 7
    cols = list(range(0,2)) + list(range(3,26))
    pca_cols = list(range(3,26))

elif which_features == "non_cnn": 
    n_components = 7
    cols = list(range(0,1)) + list(range(3,26))  + list(range(27+251,27+502,top_cadence)) + list(range(27+501+251, df.shape[1],top_cadence))
    pca_cols = list(range(3,df.shape[1],2))

else:
    n_components = 7
    cols = list(range(0,2)) + list(range(3,26))  + list(range(27+251,27+502,top_cadence)) + list(range(27+501+251, df.shape[1],top_cadence))
    pca_cols = list(range(3,df.shape[1],2))

label_col = 2

feature_cols = df.columns[cols]
print(feature_cols)

trainX, trainY, validX, validY, testX, testY = data.generateDataForERT(cols=cols, label_col=label_col, in_dir=in_dir)

number_of_trees = 500
number_of_trees = 10

all_scores = []

tss_max = 0
thresh_max = 0
metrics_max = 0

#for impurity in [0.0000025, 0.000005, 0.00001, 0.00003, 0.00006, 0.00012, 0.00025, 0.0005, 0.001, 0.002, 0.004, 0.008]:
for impurity in [0.00003, 0.00006, 0.00012, 0.00025, 0.0005, 0.001]:

    all_scores.append(ert_predict(trainX, trainY, validX, validY, impurity, None))

logger.close()

all_scores = np.array(all_scores)
tpr = all_scores[:,1] / (all_scores[:,1] + all_scores[:,4])
fpr = all_scores[:,3] / (all_scores[:,3] + all_scores[:,2])
new_metric = tpr - max(tpr) / max(fpr) * fpr
best_impurity = all_scores[np.argmax(new_metric), :]
print("Best impurity for ERT: ", best_impurity, np.max(new_metric))
print("Results on test set with CNN+ERT")

start = time.time()
impurity, tp, tn, fp, fn, tss, precision, recall, f1, fb = ert_predict(trainX, trainY, testX, testY, best_impurity[0], feature_cols)
end = time.time()
print("Time required for the prediction: ", end - start)
#impurity, tp, tn, fp, fn, tss, precision, recall, f1, fb = ert_predict(trainX, trainY, testX, testY, 0.00012)
print(int(tp), int(tn), int(fp), int(fn))

df = pd.DataFrame(all_scores, columns=['impurity', 'TP', 'TN', 'FP', 'FN', 'TSS', 'Precision', 'Recall', 'F1', 'FB'])
df.to_csv(in_dir + '/ert_impurity_val_{}.csv'.format(which_features), index=None, sep=',')

tss_max = 0
thresh_max = 0
metrics_max = 0

# Generate the CNN Only results as a function of threshold
all_scores = []

for thresh in np.arange(0.2,1,0.1):
   
    all_scores.append(cnn_predict(validY, validX[:,1], thresh))

all_scores = np.array(all_scores)
tpr = all_scores[:,1] / (all_scores[:,1] + all_scores[:,4])
fpr = all_scores[:,3] / (all_scores[:,3] + all_scores[:,2])
new_metric = tpr - max(tpr) / max(fpr) * fpr
print(new_metric)
best_threshold = all_scores[np.argmax(new_metric), :]
print("Best threshold for CNN: ", best_threshold, np.max(new_metric))
print("Results on test set with CNN")
thresh, tp, tn, fp, fn, tss, precision, recall, f1, fb = cnn_predict(testY, testX[:,1], best_threshold[0])
print(int(tp), int(tn), int(fp), int(fn))

df = pd.DataFrame(all_scores, columns=['threshold', 'TP', 'TN', 'FP', 'FN', 'TSS', 'Precision', 'Recall', 'F1', 'FB'])
df.to_csv(in_dir + '/cnn_thresholding_val.csv', index=None, sep=',')
