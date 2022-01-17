#!/bin/python

import torch
import re
import random
import os
import numpy as np
import h5py
import pandas as pd
from PIL import Image
from torchvision import transforms, utils
import torchvision.transforms.functional as F
from torchvision.transforms import Resize
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import cv2
import json

class sunspotImageDataSetLSTM(torch.utils.data.Dataset):

    def __init__(self, df, root_dir, transform, seq_length, y_col_name = 'flare'):
        
        self.name_frame = df.iloc[:, 0]
        self.label_frame = df[y_col_name].astype('long')
        self.labels = []
        self.transform = transform
        self.root_dir = root_dir
        datetime_extractor = lambda x: x.split(".")[3].split('_')
        datetimes = list(map(datetime_extractor, df['filename']))
        
        datetime_generator = lambda x: pd.to_datetime(x[0][:4] + '-' + x[0][4:6] + '-' + x[0][6:] + ' ' + x[1][:2] + ':' + x[1][2:4] + ':' + x[1][4:])
        datetimes_formatted = list(map(datetime_generator, datetimes))
        df["datetime"] = datetimes_formatted
        df = df.sort_values(by='datetime')
        
        harpnum_extractor = lambda x: int(x.split(".")[2])
        harpnums = list(map(harpnum_extractor, df['filename']))
        
        df["harpnum"] = harpnums
        df = df.sort_values(['harpnum', 'datetime'], ascending = [True, True])
        self.harpnum = df['harpnum'].values
        self.datetime = df['datetime'].values
        self.seq_length = seq_length
        self.names = []
        for index in range(len(self.name_frame)):
            properties = []
            #prev_time = self.datetime[index]
            for i in range(index, index-self.seq_length, -1):
                if i < 0 or self.harpnum[i]!=self.harpnum[index]: #or (prev_time-self.datetime[i]).astype('timedelta64[h]')>1:
                    break
                #prev_time = self.datetime[i]
                properties.insert(0, self.name_frame.iloc[i])
            if len(properties)==seq_length:
                self.names.append(list(reversed(properties)))
                self.labels.append(self.label_frame.iloc[index])

    def __len__(self):

        return len(self.names)

    def __getitem__(self, idx):

        names = self.names[idx]
        labels = self.labels[idx]

        image_sequence = np.zeros((self.seq_length,128,128))
        for i in range(len(names)):
            filename = os.path.join(self.root_dir, names[i])
            image = np.array(h5py.File(filename, 'r')['hmi']) 
            image = np.nan_to_num(image)
            image[np.where(image > 5000)] = 5000
            image[np.where(image < -5000)] = -5000
            image = (image + 5000) / 10000
            image = cv2.resize(image,(128,128))
            image = self.transform(image)
            image = np.array(image)[1,:,:].reshape((image.shape[1], image.shape[2]))
            image_sequence[i,:,:] = image

        image_sequence = image_sequence.astype('float32')
        first_filename = os.path.join(self.root_dir, names[0])
        return [first_filename, image_sequence, labels]

class sunspotImageDataSetBr(torch.utils.data.Dataset):

    def __init__(self, df, root_dir, transform):
     
        self.name_frame = df.iloc[:, 0]
        self.label_frame = df.loc[:, 'flare'].astype('long')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):

        return len(self.name_frame)

    def __getitem__(self, idx):
        
        filename = os.path.join(self.root_dir, self.name_frame.iloc[idx])
        image = np.array(h5py.File(filename, 'r')['hmi']) 
        image = np.nan_to_num(image).astype('float32')
        image[np.where(image > 5000)] = 5000
        image[np.where(image < -5000)] = -5000
        image = (image + 5000) / 10000
        image = cv2.resize(image,(128,128))
        image = self.transform(image)
        image = np.array(image)[1,:,:].reshape((-1, image.shape[1], image.shape[2]))
        labels = self.label_frame.iloc[idx]

        return [filename, image, labels]

class sunspotImageDataSet(torch.utils.data.Dataset):

    def __init__(self, df, root_dir, transform):

        self.name_frame = df.iloc[:, 0]
        self.label_frame = df.loc[:, 'flare'].astype('long')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):

        return len(self.name_frame)

    def __getitem__(self, idx):
        
        filename = os.path.join(self.root_dir, self.name_frame.iloc[idx])
        image = np.array(h5py.File(filename, 'r')['hmi']) 
        image = np.nan_to_num(image).astype('float32')
        image = np.transpose(image, (1,2,0))
        image[np.where(image > 5000)] = 5000
        image[np.where(image < -5000)] = -5000
        image = (image + 5000) / 10000
        image = cv2.resize(image,(128,128))
        image = self.transform(image)
        image = np.array(image).reshape((-1, image.shape[1], image.shape[2]))
        labels = self.label_frame.iloc[idx]

        return [filename, image, labels]

def generateTrainValidData(df, root_dir, splitType = 'by_harpnum'):
    
    with open('config.json', 'r') as jsonfile:
        config = json.load(jsonfile)
   
    # Get the forecast window, and the seed, if the splitType is by harpnum
    print(config)
    window = config['forecast_window']
    seed = config["seed"]
    
    # Create a new column that determines the flaring/non-flaring label (M+ in the next k hours)
    df['flare'] = df['M_flare_in_' + window] + df['X_flare_in_' + window]
    rows = df.loc[:,'flare'] >= 1
    df.loc[rows, 'flare'] = 1
    
    # This type of splitting is almost never used. It is incorrect, but sometimes good for debugging.
    if splitType == 'random':
        df_train = df.sample(frac=0.7, random_state=1)
    
    # Temporal split: Train 2010-2014, Valid 2015, Test 2016-2017
    elif splitType == 'temporal':
        pattern = re.compile('hmi.sharp_cea_720s\..*\.(\d\d\d\d).*')
        df_train = df[df.apply(lambda x: int(re.search(pattern, x['filename']).group(1)) <= 2014, axis=1)]
        df_test = df[df.apply(lambda x: int(re.search(pattern, x['filename']).group(1)) >= 2016, axis=1)]

        pattern = re.compile('hmi.sharp_cea_720s\.(\d+)\..*')
        harpnum = df['filename'].str.extract(pattern).astype('int64')
        harpnums = harpnum[0].unique()
        harpnum_train = df_train['filename'].str.extract(pattern).astype('int64')
        train_harpnums = harpnum_train[0].unique()
        harpnum_test = df_test['filename'].str.extract(pattern).astype('int64')
        test_harpnums = harpnum_test[0].unique()
        valid_harpnums = list(set(harpnums) - set(test_harpnums) - set(train_harpnums))

        df_train = df[df.apply(lambda x: int(re.search(pattern, x['filename']).group(1)) in train_harpnums, axis=1)] 
        df_valid = df[df.apply(lambda x: int(re.search(pattern, x['filename']).group(1)) in valid_harpnums, axis=1)]
        df_test = df[df.apply(lambda x: int(re.search(pattern, x['filename']).group(1)) in test_harpnums, axis=1)]
    
    # Split by harpnum: Train 70%, Valid 10%, Test 20%
    elif splitType == 'by_harpnum':   
        pattern = re.compile('hmi.sharp_cea_720s\.(\d+)\..*')
        harpnum = df['filename'].str.extract(pattern).astype('int64')
        harpnum_set = harpnum[0].unique()
        random.seed(seed)
        random.shuffle(harpnum_set)
        split_train = int(0.7*harpnum_set.shape[0]) 
        split_valid = int(0.85*harpnum_set.shape[0])
        train_harpnums = harpnum_set[:split_train]
        valid_harpnums = harpnum_set[split_train:split_valid]
        test_harpnums = harpnum_set[split_valid:]
        df_train = df[df.apply(lambda x: int(re.search(pattern, x['filename']).group(1)) in train_harpnums, axis=1)] 
        df_valid = df[df.apply(lambda x: int(re.search(pattern, x['filename']).group(1)) in valid_harpnums, axis=1)]
        df_test = df[df.apply(lambda x: int(re.search(pattern, x['filename']).group(1)) in test_harpnums, axis=1)]
    
    # If it should be an LSTM. 
    if config["lstm"] == 'yes':
        
        seq_length = config["seq_length"]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5,0.5), (0.5,0.5,0.5,0.5))])
        dataSetFn = sunspotImageDataSetLSTM

        #transform = transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ])
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
        trainDataSet = dataSetFn(df_train, root_dir, transform=transform, seq_length=seq_length)
        validDataSet = dataSetFn(df_valid, root_dir, transform=transform, seq_length=seq_length)
        testDataSet = dataSetFn(df_test, root_dir, transform=transform, seq_length=seq_length)
        return [trainDataSet, validDataSet, testDataSet]

    # Decide which dataset to use based on the number of channels. If it is a single channel, then use the Br component, 
    # else use all.
    if config["input_channels"] == 1:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
        dataSetFn = sunspotImageDataSetBr
    else:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5,0.5), (0.5,0.5,0.5,0.5))])
        dataSetFn = sunspotImageDataSet

    trainDataSet = dataSetFn(df_train, root_dir, transform=transform)
    validDataSet = dataSetFn(df_valid, root_dir, transform=transform)
    testDataSet = dataSetFn(df_test, root_dir, transform=transform)
    return [trainDataSet, validDataSet, testDataSet]

# Function to generate datasets to be used with ERTs.
def generateDataForERT(cols, label_col, in_dir):
    
    # This function usually is executed after the CNN Only runs are complete. The train and test dataset files 
    # present in "in_dir" contain the combination of the CNN Only probability outputs together with the 
    # SHARPs and Topological feature sets. These will be available once you run vgg16.py and model_evaluator.py.

    df_train = pd.read_csv(in_dir + '/train.csv')
    df_valid = pd.read_csv(in_dir + '/val.csv')
    df_test = pd.read_csv(in_dir + '/test.csv')
    
    # Remove all infinities and nan values

    df_train = df_train.replace([np.inf, -np.inf], np.nan)
    df_train = df_train.dropna()
    df_valid = df_valid.replace([np.inf, -np.inf], np.nan)
    df_valid = df_valid.dropna()
    df_test = df_test.replace([np.inf, -np.inf], np.nan)
    df_test = df_test.dropna()
    
    # Normalize the features except for the CNN probability feature
    # on both training and valid sets.

    y_col_train = df_train.iloc[:,label_col]
    y_col_valid = df_valid.iloc[:,label_col]
    y_col_test = df_test.iloc[:,label_col]

    cols_to_norm = df_train.columns[cols[2:]]
    data = df_train[cols_to_norm].values
    scaler = StandardScaler()
    scaler.fit(data)

    data = scaler.transform(data)
    df_new = pd.DataFrame(data, columns=[x for x in cols_to_norm])
    df_new.insert(loc=0, column="cnn_prob", value=df_train["cnn_prob"].values)
    df_new.insert(loc=0, column="filename", value=df_train["filename"].values)
    df_train = df_new
    
    data = df_valid[cols_to_norm].values
    data = scaler.transform(data)
    df_new = pd.DataFrame(data, columns=[x for x in cols_to_norm])
    df_new.insert(loc=0, column="cnn_prob", value=df_valid["cnn_prob"].values)
    df_new.insert(loc=0, column="filename", value=df_valid["filename"].values)
    df_valid = df_new

    data = df_test[cols_to_norm].values
    data = scaler.transform(data)
    df_new = pd.DataFrame(data, columns=[x for x in cols_to_norm])
    df_new.insert(loc=0, column="cnn_prob", value=df_test["cnn_prob"].values)
    df_new.insert(loc=0, column="filename", value=df_test["filename"].values)
    df_test = df_new
    
    trainX, trainY = np.array(df_train), np.array(y_col_train)
    validX, validY = np.array(df_valid), np.array(y_col_valid)
    testX, testY = np.array(df_test), np.array(y_col_test)
     
    #trainX, trainY = np.array(df_train.iloc[:,cols]), np.array(df_train.iloc[:,label_col])
    #validX, validY = np.array(df_valid.iloc[:,cols]), np.array(df_valid.iloc[:,label_col])

    return [trainX, trainY, validX, validY, testX, testY]

# Data loader function.
def get_loader(set, sampler, batch_size):    
    sunspotLoader = torch.utils.data.DataLoader(set, sampler=sampler, num_workers=2, batch_size=batch_size, shuffle=True)
    return sunspotLoader
    
    trainX, trainY = np.array(df_train), np.array(y_col_train)
    validX, validY = np.array(df_valid), np.array(y_col_valid)
    
    #trainX, trainY = np.array(df_train.iloc[:,cols]), np.array(df_train.iloc[:,label_col])
    #validX, validY = np.array(df_valid.iloc[:,cols]), np.array(df_valid.iloc[:,label_col])

    return [trainX, trainY, validX, validY]

# Data loader function.
def get_loader(set, sampler, batch_size):    
    sunspotLoader = torch.utils.data.DataLoader(set, sampler=sampler, num_workers=2, batch_size=batch_size, shuffle=True)
    return sunspotLoader
