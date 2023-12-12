import subprocess
import os
from Dataset.ReadyToTrain_DS import Img_Dataset
from Dataset.ReadyToTrain_DS import get_DataLoaders
from Dataset.Transforms import get_transforms
from Dataset import Unzip_DS
from Training.Train_DomainOnly import *

from Models.Loss_Functions import FocalLoss

# Once datasets have been downloaded (Using DS_Download.sh) you can unzip them
Unzip_DS.UnzipFolders("Tanzania")
Unzip_DS.UnzipFolders("IvoryCoast")

# Hyperparameters
batch_size = 8
number_epochs = 4
learning_rate = 0.1
starter_channels = 8
momentum = 0
bilinear = True
loss_function = FocalLoss(gamma = 1.5)
device = get_training_device()
transforms = get_transforms()
VI = False
normalization = 'Linear_1_99'

# Build Dataset class
train_loader, val_loader, test_loader = get_DataLoaders('TanzaniaSplit1', batch_size, transforms, normalization, VI)

n_channels = next(enumerate(train_loader))[1][0].shape[1]
n_classes = 2

# Define the network
network = UNet(n_channels, n_classes,  bilinear, starter_channels, up_layer = 4)

# Train the model
f1_val, network_trained = training_loop(network, train_loader, val_loader, learning_rate, starter_channels, momentum, number_epochs, loss_function)

# Evaluate the model
f1_test, loss_test = evaluate(network_trained, test_loader, loss_function, BinaryF1Score(), Love = False)



