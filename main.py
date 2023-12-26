import subprocess
import os
import time
from torchmetrics.classification import BinaryF1Score, JaccardIndex
import pandas as pd

from Dataset.ReadyToTrain_DS import Img_Dataset, get_DataLoaders, get_LOVE_DataLoaders
from Dataset.Transforms import get_transforms
from Dataset import Unzip_DS
from Training.Train_DomainOnly import *
from Training.Hyper_Parameter_Tuning import *

from Models.Loss_Functions import FocalLoss

# Once datasets have been downloaded (Using DS_Download.sh) you can unzip them
Unzip_DS.UnzipFolders("Tanzania")
Unzip_DS.UnzipFolders("IvoryCoast")

# If needed run Hyperparameter tuning to get the optimal HPs
# BS = [8]
# LR = [0.5, 1, 2]
# decay = [0.8, 1]
# STCh = [16]
# MU = [0]
# Bi = [True]
# gamma = [2, 4, 6]
# VI = [True]

# for i in range(3):
#     HP_Tuning('TanzaniaSplit'+str(i+1), BS, LR, STCh, MU, Bi, gamma, VI, decay, tr_size = 0.15, val_size = 0.95)
#     HP_Tuning('IvoryCoastSplit'+str(i+1), BS, LR, STCh, MU, Bi, gamma, VI, decay, tr_size = 0.15, val_size = 0.5)


# Hyperparameters for DOMAIN ONLY TRAINING
domain = 'IvoryCoast'
## Related to DS
batch_size = 8
transforms = get_transforms()
normalization = 'Linear_1_99'
VI = False

## Related to the network
n_classes = 2
bilinear = True
starter_channels = 16
up_layer = 4

## Related to training and evaluation
number_epochs = 15
learning_rate = 2
momentum = 0.0
loss_function = FocalLoss(gamma = 2)
accu_function = BinaryF1Score()
device = get_training_device()

DS_args = [batch_size, transforms, normalization, VI]
network_args = [n_classes, bilinear, starter_channels, up_layer]
training_args = [learning_rate, momentum, number_epochs, loss_function]
eval_args = [loss_function, accu_function]

Stats = train_3fold_DomainOnly(domain, DS_args, network_args, training_args, eval_args)


# For LoveDA
# train_loader, val_loader, test_loader = get_LOVE_DataLoaders(['urban'], batch_size = batch_size)

# n_channels = next(enumerate(train_loader))[1]['image'].shape[1] #get band number fomr actual data
# n_classes = 2

# # Define the network
# network = UNet(n_channels, n_classes,  bilinear, starter_channels, up_layer = 4)

# # Train the model
# print("Starting training...")
# start = time.time()
# f1_val, network_trained, spearman = training_loop(network, train_loader, val_loader, learning_rate, starter_channels, momentum, number_epochs, loss_function, accu_function=JaccardIndex(task = 'multiclass', num_classes = n_classes) , Love = True)
# print("Network trained. Took ", round(time.time() - start, 0), 's\n')

# # Evaluate the model
# f1_test, loss_test = evaluate(network_trained, test_loader, loss_function, accu_function=JaccardIndex(task = 'multiclass', num_classes = n_classes), Love = True)

# print("JACCARD_Validation:", f1_val)
# print("JACCARD_Test:      ", f1_test)