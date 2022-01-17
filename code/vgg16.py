#!/usr/bin/env python
import numpy as np
import torchvision
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, utils
from torchvision import *
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import os
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import importlib
import data
from data import get_loader, generateTrainValidData
import json 
from losses import FocalLoss
import torch.optim as optim
from cnn_models import Vgg16, Vgg16LSTM, Vgg16Conv3D
import time

def trainNet(net, device, batch_size, n_epochs, sunspotTrainSet, sunspotValidSet, config):
    
    weight_decay = config["weight_decay"] 
    learning_rate = config["learning_rate"]
    gamma = config["gamma"]
    alpha_inv = config["alpha_inv"]

    #Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("device=", device)
    print("=" * 30)
    
    
    # Create data loaders for the training and validation set
    train_loader = get_loader(sunspotTrainSet, sampler=None, batch_size=batch_size)
    val_loader = get_loader(sunspotValidSet, sampler=None, batch_size=batch_size)
    
    # Get the number of batches
    n_batches = len(train_loader)
    
    # We use the Focal Loss, which is a better version of the BCE Loss.
    # Read https://amaarora.github.io/2020/06/29/FocalLoss.html for more details. 
    # The two important hyperparameters here are alpha and gamma. alpha is used for
    # weighting the two classes: [alpha, 1-alpha]. To get the weight 1:10, set 
    # alpha_inv = 1/11. gamma is an additional exponent term used in the focal loss
    # to allow the model to be less confident about predicting the ground truth, 
    # and therefore be less "overconfident". gamma=0 reduces this to the usual 
    # BCE Loss.

    loss = FocalLoss(alpha=1/alpha_inv, gamma=gamma, size_average=False)

    # We use the Adagrad optimizer. Adam has known to not work so well. The two
    # important parameters are learning_rate and weight_decay (L2 regularization
    # constant). 
    
    optimizer = optim.Adagrad(net.parameters(), lr=learning_rate, weight_decay=weight_decay, lr_decay=1e-4)
    
    # The learning rate is adjusted using Cosine Annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    #Time for printing
    training_start_time = time.time()
    
    train_plot = []
    valid_plot = []
    tss_plot = []
    tp_plot = []
    fp_plot = []
    tn_plot = []
    fn_plot = []

    total_val_loss_save = np.inf
    #Loop for n_epochs
    for epoch in range(n_epochs):
        
        running_loss = 0.0
        print_every = n_batches // 10
        start_time = time.time()
        total_train_loss = 0
       
        # This allows the network to be updated.
        net.train()

        for i, data in enumerate(train_loader, 0):

            #Get inputs
            _, inputs, labels = data
            
            #Wrap them in a Variable object
            inputs, labels = Variable(inputs), Variable(labels)
            inputs, labels = inputs.to(device), labels.to(device)
            
            #Set the parameter gradients to zero
            optimizer.zero_grad()
            
            # Forward pass, backward pass, optimize. This calls the forward() 
            # function
            outputs = net(inputs)

            # Compute the loss, and backpropagate. The backward() function
            # computes the derivatives at each layer in the network. 
            # The step function will actually adjust all the weights.

            # outputs = torch.nn.Softmax(dim=1)(outputs)
            loss_size = loss(outputs, labels)
            loss_size.backward()
            optimizer.step()

            #Print statistics
            running_loss += loss_size.data.item()
            total_train_loss += loss_size.data.item()
            
            #Print every 10th batch of an epoch
            if (i + 1) % (print_every + 1) == 0:
                print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                        epoch+1, int(100 * (i+1) / n_batches), running_loss / print_every, time.time() - start_time))
                # Reset running loss and time
                running_loss = 0.0
                start_time = time.time()
        
        #### Training for 1 epoch done ########

        # At the end of the epoch, do a pass on the validation set
        total_val_loss = 0
        confusion_matrix = torch.zeros(2, 2)
        correct = 0
        total = 0
        
        # This freezes the network weights during validation.
        # Using torch.no_grad() is faster. Change later ...

        net.eval()
        y_prob = []
        y_true = []

        for _, inputs, labels in val_loader:
            
            #Wrap tensors in Variables
            inputs, labels = Variable(inputs), Variable(labels)
            inputs, labels = inputs.to(device), labels.to(device)
    
            #Forward pass
            val_outputs = net(inputs)
            #val_outputs = torch.nn.Softmax(dim=1)(val_outputs)
            _, predicted = torch.max(val_outputs.data, 1)        
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

            val_loss_size = loss(val_outputs, labels)
            total_val_loss += val_loss_size.data.item()
            
            y_prob += val_outputs.data[:,1].tolist()
            y_true += labels.tolist()

        tp = confusion_matrix[1,1]
        tn = confusion_matrix[0,0]
        fp = confusion_matrix[0,1]
        fn = confusion_matrix[1,0]
        
        # Varad: Don't we need a softmax for this?
        aps = average_precision_score(y_true, y_prob)

        print("Validation loss = {:.2f}".format(total_val_loss / len(val_loader)))
        print("TSS = {:.2f}".format((tp) / (tp + fn) - (fp) / (fp + tn)))
        print("Average Precision Score = {:.2f}".format(aps))
        print("TP = {}, FP = {}, FN = {} TN  = {}".format(tp, fp, fn, tn))
        print("Training finished, took {:.2f}s".format(time.time() - training_start_time))

        total_val_loss_save = total_val_loss / len(val_loader)
        #model_save = net

        train_plot.append(total_train_loss / len(train_loader))
        valid_plot.append(total_val_loss / len(val_loader))
        tss_plot.append((tp) / (tp + fn) - (fp) / (fp + tn))
        tp_plot.append(tp)
        fp_plot.append(fp)
        tn_plot.append(tn)
        fn_plot.append(fn)
        
        # Update the learning rate
        scheduler.step() 
    
    # Done training, save all the metrics and the error profiles to the out_dir

    train_plot = np.array(train_plot).reshape((-1,1))
    valid_plot = np.array(valid_plot).reshape((-1,1))
    tss_plot = np.array(tss_plot).reshape((-1,1))
    tp_plot = np.array(tp_plot).reshape((-1,1))
    fp_plot = np.array(fp_plot).reshape((-1,1))
    tn_plot = np.array(tn_plot).reshape((-1,1))
    fn_plot = np.array(fn_plot).reshape((-1,1))
    
    outdir = config["output_dir"]
    np.save(outdir + '/stats', np.concatenate((train_plot, valid_plot, tss_plot, tp_plot, fp_plot, tn_plot, fn_plot), axis = 1)) 
    plt.figure()
    plt.plot(train_plot, label = 'train')
    plt.plot(valid_plot, label = 'valid')
    plt.plot(tss_plot, label = 'tss')
    plt.legend()
    plt.savefig(outdir + '/errors.png')

# Evaluating the model on the test set.
def testNet(net, device, sunspotTestSet):
    
    correct = 0
    total = 0
    y_prob = []
    y_true = []

    test_loader = get_loader(sunspotTestSet, sampler=None, batch_size=128)
    confusion_matrix = torch.zeros(2, 2)

    with torch.no_grad():
        for data in test_loader:
            #Get inputs
            _, inputs, labels = data
                
            #Wrap them in a Variable object
            inputs, labels = Variable(inputs), Variable(labels)
            inputs, labels = inputs.to(device), labels.to(device)

            # Get the prediction and softmax it.
            outputs = net(inputs)
            outputs = torch.nn.Softmax(dim=1)(outputs)

            # Get the maximum probability on axis 1 to determine the label.
            # The first argument returned is the value, which we can ignore.
            _, predicted = torch.max(outputs.data, 1)        

            # Determine the total samples in the batch, and how many were 
            # correctly labeled. Also compute the confusion matrix. 

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

            y_prob += outputs.data[:,1].tolist()
            y_true += labels.tolist()
        
        # Compute metrics
        aps = average_precision_score(y_true, y_prob)
        fpr, tpr, thresholds = roc_curve(np.array(y_true), np.array(y_prob))
        roc_auc = auc(fpr, tpr)

        precision, recall, thresholds = precision_recall_curve(np.array(y_true), np.array(y_prob))
        pr_auc = auc(recall, precision)

    print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))
    print('ROC AUC = ', roc_auc)
    print('PR AUC = ', pr_auc)
    print('Average Precision Score', aps)

    print(confusion_matrix)

    tp = confusion_matrix[1,1]
    tn = confusion_matrix[0,0]
    fp = confusion_matrix[0,1]
    fn = confusion_matrix[1,0]

    print((tp) / (tp + fn) - (fp) / (fp + tn))

    return [y_true, y_prob]

def main():
    
    # Load the configuration parameters for the experiment, split = {'random', 'by_harpnum', 'temporal'}
    with open('config.json', 'r') as jsonfile:
        config = json.load(jsonfile)
    
    outdir = config["output_dir"]
    splitType = config["split"]

    # Sanity check to make sure that we are not overwriting an existing output configuration 
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    elif len(os.listdir(outdir)) != 0:
        print("Output directory {} already exists. You could be overwriting. Delete the contents before continuing.".format(outdir))
        exit(-1)
    
    # Copy the configuration file into outdir
    with open(outdir + '/config.json', 'w') as outfile:
        json.dump(config, outfile)
    
    # Decide which device to run on
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    # Read in the labels file, and generate the train, valid, test datasets
    df = pd.read_csv(config["labels_file"], sep=",", header='infer')
    sunspotTrainSet, sunspotValidSet, sunspotTestSet = generateTrainValidData(df, root_dir='/', splitType=splitType)

    # Initialize the model. And map it to the device.
    if config["lstm"] == "yes":
        net = Vgg16LSTM()
    else:
        net = Vgg16(config["input_channels"])
        #net = Vgg16Conv3D(config["input_channels"])

    net.to(device)
    
    # Start the model training. 
    trainNet(net=net, device=device, batch_size=256, n_epochs=10, sunspotTrainSet=sunspotTrainSet, sunspotValidSet=sunspotValidSet, config=config)

    # Save the model to the out directory. 
    torch.save(net.state_dict(), outdir + '/model')

    # Perform the model validation. 
    [y_true, y_prob] = testNet(net=net, device=device, sunspotTestSet=sunspotValidSet) 

    np.save(outdir + '/val_true', np.array(y_true))
    np.save(outdir + '/val_pred', np.array(y_prob))

    # Start the model testing. 
    [y_true, y_prob] = testNet(net=net, device=device, sunspotTestSet=sunspotTestSet) 

    np.save(outdir + '/true', np.array(y_true))
    np.save(outdir + '/pred', np.array(y_prob))


if __name__ == "__main__":
    
    main()
