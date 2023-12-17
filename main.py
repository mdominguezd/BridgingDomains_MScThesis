import subprocess
import os
import time
from torchmetrics.classification import BinaryF1Score

from Dataset.ReadyToTrain_DS import Img_Dataset
from Dataset.ReadyToTrain_DS import get_DataLoaders
from Dataset.Transforms import get_transforms
from Dataset import Unzip_DS
from Training.Train_DomainOnly import *
from Training.Hyper_Parameter_Tuning import *
from Models.U_Net import UNet

from Models.Loss_Functions import FocalLoss

# Once datasets have been downloaded (Using DS_Download.sh) you can unzip them
print("Unziping downloaded data...")
Unzip_DS.UnzipFolders("Tanzania")
Unzip_DS.UnzipFolders("IvoryCoast")
print("Folders have been unzipped.\n")

# If needed run Hyperparameter tuning to get the optimal HPs
BS = [4,8]
LR = [0.1, 0.01]
STCh = [8, 16]
MU = [0, 0.2]
Bi = [True, False]
gamma = [0.7, 1.5]
VI = [True, False]

HP_Tuning('TanzaniaSplit1', BS, LR, STCh, MU, Bi, gamma, VI)

# Hyperparameters
batch_size = 6
number_epochs = 6
learning_rate = 0.1
starter_channels = 8
momentum = 0
bilinear = True
loss_function = FocalLoss(gamma = 1.5)
device = get_training_device()
transforms = get_transforms()
VI = False
normalization = 'Linear_1_99'

folds = 3

# For 3-fold Cross-Validation
for i in range(folds):
    
    # Build Dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = get_DataLoaders('TanzaniaSplit'+str(i+1), batch_size, transforms, normalization, VI)
    print("Dataloaders created.\n")
    
    n_channels = next(enumerate(train_loader))[1][0].shape[1] #get band number fomr actual data
    n_classes = 2
    
    # Define the network
    network = UNet(n_channels, n_classes,  bilinear, starter_channels, up_layer = 4)
    
    # Train the model
    print("Starting training...")
    start = time.time()
    f1_val, network_trained = training_loop(network, train_loader, val_loader, learning_rate, starter_channels, momentum, number_epochs, loss_function)
    print("Network trained. Took ", round(time.time() - start, 0), 's\n')
    
    # Evaluate the model
    f1_test, loss_test = evaluate(network_trained, test_loader, loss_function, BinaryF1Score(), Love = False)
    
    print("F1_Validation:", f1_val)
    print("F1_Test:      ", f1_test)

