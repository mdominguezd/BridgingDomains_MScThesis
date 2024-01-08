import subprocess
import os
import time
from torchmetrics.classification import BinaryF1Score, JaccardIndex
import pandas as pd

from Dataset.ReadyToTrain_DS import *
from Dataset.Transforms import *
from Dataset.Visualize_DataSet import *
from Dataset import Unzip_DS
from Training.Train_DomainOnly import *
from Training.Train_DANN import *
from Training.Hyper_Parameter_Tuning import *
from Validate.figures_for_validation import *
from Validate.predict import *

from Models.Loss_Functions import FocalLoss

####################################################################
######################## For Own Dataset ###########################
####################################################################

# # Once datasets have been downloaded (Using DS_Download.sh) you can unzip them
# Unzip_DS.UnzipFolders("Tanzania")
# Unzip_DS.UnzipFolders("IvoryCoast")

# # If needed run Hyperparameter tuning to get the optimal HPs
# BS = [8]
# LR = [0.5, 1.5, 3]
# decay = [0.8]
# STCh = [16]
# MU = [0]
# Bi = [True]
# gamma = [1, 2, 4]
# VI = [True]

# for i in range(3):
#     HP_Tuning('TanzaniaSplit'+str(i+1), BS, LR, STCh, MU, Bi, gamma, VI, decay, tr_size = 0.1, val_size = 0.95)
#     HP_Tuning('IvoryCoastSplit'+str(i+1), BS, LR, STCh, MU, Bi, gamma, VI, decay, tr_size = 0.1, val_size = 0.5)

# plot_HyperparameterTuning()
    
# # Hyperparameters for DOMAIN ONLY TRAINING
# domain = 'Tanzania'
# ## Related to DS
# batch_size = 8
# transforms = get_transforms()
# normalization = 'Linear_1_99'
# VI = False

# ## Related to the network
# n_classes = 2
# bilinear = True
# starter_channels = 16
# up_layer = 4

# ## Related to training and evaluation
# number_epochs = 15
# learning_rate = 0.5
# momentum = 0.0
# loss_function = FocalLoss(gamma = 2)
# accu_function = BinaryF1Score()
# device = get_training_device()

# DS_args = [batch_size, transforms, normalization, VI]
# network_args = [n_classes, bilinear, starter_channels, up_layer]
# training_args = [learning_rate, momentum, number_epochs, loss_function]
# eval_args = [loss_function, accu_function]

# Stats = train_3fold_DomainOnly(domain, DS_args, network_args, training_args, eval_args)

#### DOMAIN ADAPTATION ####
source_domain = 'IvoryCoastSplit1'
target_domain = 'TanzaniaSplit1'

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
attention = False

DS_args = [batch_size, transforms, normalization, VI, True]
network_args = [n_classes, bilinear, starter_channels, up_layer, attention]

lr = 0.005
momentum = 0.4
epochs = 5
Love = False
seg_loss_function = FocalLoss(gamma = 0.7)
domain_loss_function = torch.nn.BCEWithLogitsLoss()
accu_function = BinaryF1Score()

best_model_accuracy, best_network = DANN_training_loop(source_domain, target_domain, DS_args, network_args, lr, momentum, epochs, Love, seg_loss_function, domain_loss_function, accu_function)

####################################################################
########################### For LoveDA #############################
####################################################################

# If needed run Hyperparameter tuning to get the optimal HPs
# domain = ['urban', 'rural']
# BS = [4]
# LR = [1.5, 3]
# STCh = [16, 8]
# MU = [0, 0.5]
# Bi = [False, True]
# gamma = [2, 4]
# decay = [0.8, 1]

# LoveDA_HP_Tuning(domain, BS, LR, STCh, MU, Bi, gamma, decay, tr_size = 0.1)

#### DOMAIN ONLY ####

# Hyperparameters for DOMAIN ONLY TRAINING
# domain = ['rural']

# ## Related to DS
# batch_size = 4
# transforms = get_transforms()

# ## Related to the network
# n_classes = 8
# bilinear = True
# starter_channels = 16
# up_layer = 4
# attention = True

# ## Related to training and evaluation
# learning_rate = 2.5
# momentum = 0.0
# number_epochs = 5
# loss_function = FocalLoss(gamma = 4, ignore_index = 0)
# accu_function = JaccardIndex(task = 'multiclass', num_classes = n_classes)
# Love = True
# device = get_training_device()

# # Group all arguments in three lists
# DS_args = [batch_size, transforms]
# network_args = [n_classes,  bilinear, starter_channels, up_layer, attention]
# training_loop_args = [learning_rate, momentum, number_epochs, loss_function, accu_function, Love]

# accu, network_trained = train_LoveDA_DomainOnly(domain, DS_args, network_args, training_loop_args)

#### DOMAIN ADAPTATION ####
# source_domain = ['urban']
# target_domain = ['rural']

# ## Related to DS
# batch_size = 2
# transforms = get_transforms()

# ## Related to the network
# n_classes = 8
# bilinear = True
# starter_channels = 16
# up_layer = 4
# attention = False
# Love = True

# DS_args = [batch_size, transforms, True]
# network_args = [n_classes, bilinear, starter_channels, up_layer, attention, Love]

# lrs = [0.1, 0.01, 0.001, 0.0001]

# for lr in lrs:
#     momentum = 0
#     epochs = 10
#     seg_loss_function = FocalLoss(gamma = 0.7, ignore_index = 0)
#     domain_loss_function = torch.nn.BCEWithLogitsLoss()
#     accu_function = JaccardIndex(task = 'multiclass', num_classes = n_classes, ignore_index = 0)
    
#     best_model_accuracy, best_network = DANN_training_loop(source_domain, target_domain, DS_args, network_args, lr, momentum, epochs, Love, seg_loss_function, domain_loss_function, accu_function)

#     print(lr, best_model_accuracy)

#### PREDICTIONS ####

# predict_LoveDA('BestDANNModel.pt', ['urban', 'rural'])
