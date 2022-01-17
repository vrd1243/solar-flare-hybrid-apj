#!/bin/python
import sys
import numpy as np
import torchvision
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, utils
from torchvision import *
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import os
from PIL import Image
import data 
from matplotlib import pyplot as plt
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import importlib
import json
from cnn_models import Vgg16, Vgg16Conv3D

# The output directory of the CNN Only results is the command line argument.
# The saved model state and the config file will be loaded from here.
outdir = sys.argv[1]

with open(outdir + '/config.json', 'r') as jsonfile:
    config = json.load(jsonfile)

weight_decay_global = config["weight_decay"] 
learning_rate_global = config["learning_rate"]
gamma_global = config["gamma"]
alpha_inv = config["alpha_inv"]
outdir = config["output_dir"]
splitType = config["split"]

if not os.path.exists(outdir):
    print("Output directory doesn not exist.")
    exit(-1)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
df = pd.read_csv(config["labels_file"], sep=",", header='infer')

# Load the data
sunspotTrainSet, sunspotValidSet, sunspotTestSet = data.generateTrainValidData(df, root_dir='/', splitType=splitType)

# Generate the data loaders
batch_size=128
train_loader = data.get_loader(sunspotTrainSet, sampler=None, batch_size=batch_size)
val_loader = data.get_loader(sunspotValidSet, sampler=None, batch_size=batch_size)
test_loader = data.get_loader(sunspotTestSet, sampler=None, batch_size=batch_size)

# Instantiate and initialize the model
#vgg16 = Vgg16(config["input_channels"])
vgg16 = Vgg16Conv3D(config["input_channels"])
vgg16.load_state_dict(torch.load(outdir + '/model'))
net = vgg16
net.to(device)

# Generate the metrics for each of the datasets (train, test). 
# For each sample in the set, the filename, the CNN output probability 
# and the actual label are returned.
def generate_results(loader):
    
    correct = 0
    total = 0
    pred = []
    true = []
    names = []

    confusion_matrix = torch.zeros(2, 2)
    with torch.no_grad():
        for data in loader:
            #Get inputs
            filenames, inputs, labels = data
                
            #Wrap them in a Variable object
            inputs, labels = Variable(inputs), Variable(labels)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            outputs = torch.nn.Softmax(dim=1)(outputs)
            _, predicted = torch.max(outputs.data, 1)        
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

            pred += outputs.data[:,1].tolist()
            true += labels.tolist()
            names += filenames

        aps = average_precision_score(true, pred)
        fpr, tpr, thresholds = roc_curve(np.array(true), np.array(pred))
        roc_auc = auc(fpr, tpr)

        precision, recall, thresholds = precision_recall_curve(np.array(true), np.array(pred))
        pr_auc = auc(recall, precision)

        print('Accuracy of the network on the set images: %d %%' % (
            100 * correct / total))
        print('ROC AUC = ', roc_auc)
        print('PR AUC = ', pr_auc)
        print('Average Precision Score', aps)

        return [true, pred, names]

# Generate the CNN outputs for the training and the test set and prepare a dataframe
[train_true, train_pred, train_names] = generate_results(train_loader)
[val_true, val_pred, val_names] = generate_results(val_loader)
[test_true, test_pred, test_names] = generate_results(test_loader)

train_df = pd.DataFrame(np.array([train_names, train_pred, train_true]).T, columns=['filename', 'cnn_prob', 'true'])
val_df = pd.DataFrame(np.array([val_names, val_pred, val_true]).T, columns=['filename', 'cnn_prob', 'true'])
test_df = pd.DataFrame(np.array([test_names, test_pred, test_true]).T, columns=['filename', 'cnn_prob', 'true'])

# Determine the active region ID from the filename column. This includes the SHARP # and timestamp.
# This is stored in the AR column
train_df['AR'] = train_df['filename'].str.extract('.*(hmi.*TAI)')
test_df['AR'] = test_df['filename'].str.extract('.*(hmi.*TAI)')
val_df['AR'] = val_df['filename'].str.extract('.*(hmi.*TAI)')

# Do the same for the database containing the SHARPs and topological features.
hmi_top = pd.read_csv('/srv/data/features/hmi_all_top.csv')
hmi_top['AR'] = hmi_top['filename'].str.extract('.*(hmi.*TAI)')
hmi_top = hmi_top.drop(['filename'], axis=1)

# Merge the numerical features with the CNN probability
train_df = train_df.merge(hmi_top, left_on='AR', right_on='AR')
test_df = test_df.merge(hmi_top, left_on='AR', right_on='AR')
val_df = val_df.merge(hmi_top, left_on='AR', right_on='AR')

# Remove the AR column. It was only required for merging
train_df = train_df.drop(['AR'], axis=1)
val_df = val_df.drop(['AR'], axis=1)
test_df = test_df.drop(['AR'], axis=1)

# Write to output files. 
train_df.to_csv(outdir + '/train.csv', index=None)
val_df.to_csv(outdir + '/val.csv', index=None)
test_df.to_csv(outdir + '/test.csv', index=None)
