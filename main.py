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
from Validate.domain_adaptation_performance import *

from Models.Loss_Functions import FocalLoss

####################################################################
######################## For Own Dataset ###########################
####################################################################

# # # Once datasets have been downloaded (Using DS_Download.sh) you can unzip them
# Unzip_DS.UnzipFolders("Tanzania")
# Unzip_DS.UnzipFolders("IvoryCoast")

# If needed run Hyperparameter tuning to get the optimal HPs
# Hyper_Parameter_Tuning.run_HP_Tuning()
    
# # Hyperparameters for DOMAIN ONLY TRAINING
# Train_DomainOnly.run_DomainOnly(domain = 'Tanzania')
# Train_DomainOnly.run_DomainOnly(domain = 'IvoryCoast')

# For Features extracted analysis
# S_, T_ = get_features_extracted('IvoryCoastSplit1', 'TanzaniaSplit1', 'OverallBestModel'+domain+'.pt', DS_two_doms, Love = False)

# _ = tSNE_source_n_target(S_, T_)

# cos = cosine_sim(S_, T_)
# euc = euc_dist(S_, T_)

# print('cosine simmilarity:', cos, '\neuclidean distance:', euc)

#### DOMAIN ADAPTATION ####

### HP Tuning
# source_domain = 'IvoryCoastSplit1'
# target_domain = 'TanzaniaSplit1'

# BS = [8]
# LR_s = [0.00001, 0.0001, 0.001]
# LR_d = [None]
# STCh = [16]
# MU = [None]
# Bi = [True]
# gamma = [2, 4]
# VI = [True]
# atts = [True]
# e_0s = [25, 40, 55]
# l_max = [0.1, 0.3]
# up_layers = [4]
# tr_size = 0.1
# val_size = 0.1

# DANN_HP_Tuning(source_domain, target_domain, BS, LR_s, LR_d, STCh, MU, Bi, gamma, VI, atts, e_0s, l_max, up_layers, tr_size, val_size)

# TUNED MODEL
source_domain = 'IvoryCoastSplit1'
target_domain = 'TanzaniaSplit1'

## Related to DS
batch_size = 16
transforms = get_transforms()
normalization = 'Linear_1_99'
VI = True
DA = True

DS_args = [batch_size, transforms, normalization, VI, DA, None, None]

## Related to the network
n_classes = 2
bilinear = True
sts = 16
up_layer = 4
att = True

network_args = [n_classes, bilinear, sts, up_layer, att]

lr_s = 0.0001
lr_d = 0.0001

optim_args = [lr_s, lr_d]

epochs = 60
e_0 = 30
l_max = 0.01

DA_args = [epochs, e_0, l_max]

gamma = 4

seg_loss = FocalLoss(gamma = gamma)

Love = False
binary_love = False

domain_loss_function = torch.nn.BCEWithLogitsLoss(reduction = 'mean')
accu_function = BinaryF1Score()


best_model_acc, target_acc, best_seg_disc_acc, best_network, training_list = DANN_training_loop(source_domain, target_domain, DS_args, network_args, optim_args, DA_args, seg_loss = seg_loss, domain_loss = domain_loss_function, semi = True, semi_perc = 0.5)


# ## Visualize DANN
# tr, val, test = get_DataLoaders('TanzaniaSplit1', 4, None, 'Linear_1_99', True)

# plot_GTvsPred_sample_images('BestDANNModel.pt', test, 4)

# tr, val, test = get_DataLoaders('IvoryCoastSplit1', 4, None, 'Linear_1_99', True)

# plot_GTvsPred_sample_images('BestDANNModel.pt', test, 4)

# DS_args = [batch_size, transforms, normalization, VI, False, 0.1, 0.5]
# raw = tSNE_source_n_target(source_domain, target_domain, 'BestDANNModel.pt', DS_args)

####################################################################
########################### For LoveDA #############################
####################################################################

# If needed run Hyperparameter tuning to get the optimal HPs
# domain = ['urban']
# BS = [4]
# LR = [0.001, 0.01, 0.1, 1]
# STCh = [16]
# MU = [0, 0.2]
# Bi = [True]
# gamma = [2]
# decay = [0.8]
# atts = [True, False]
# res = [False]

# LoveDA_HP_Tuning(domain, BS, LR, STCh, MU, Bi, gamma, decay, atts, res, tr_size = 0.05)

### DOMAIN ONLY ####

# Hyperparameters for DOMAIN ONLY TRAINING
# domain = ['rural']

# ## Related to DS
# batch_size = 16
# transforms = None # For LoveDA they are made on the fly
# DA = False

# ## Related to the network
# n_classes = 8
# bilinear = True
# starter_channels = 16
# up_layer = 4
# attention = True
# resunet = False

# ## Related to training and evaluation
# learning_rate = 0.1
# momentum = 0.0
# number_epochs = 70
# loss_function = FocalLoss(gamma = 3, ignore_index = 0)
# accu_function = JaccardIndex(task = 'multiclass', num_classes = n_classes, ignore_index = 0)
# Love = True
# binary_love = False
# device = get_training_device()

# # Group all arguments in three lists
# DS_args = [batch_size, transforms, DA, None, None]
# network_args = [n_classes,  bilinear, starter_channels, up_layer, attention, resunet]
# training_loop_args = [learning_rate, momentum, number_epochs, loss_function, accu_function, Love, binary_love]

# accu, network_trained = train_LoveDA_DomainOnly(domain, DS_args, network_args, training_loop_args)

# print(accu)

# tr, val, test = get_LOVE_DataLoaders(['urban'], 2, transforms)

# plot_GTvsPred_sample_images('BestModel.pt', tr, 2, Love = True, binary_love = binary_love, DA = False)

# tr, val, test = get_LOVE_DataLoaders(['rural'], 2, transforms)

# plot_GTvsPred_sample_images('BestModel.pt', tr, 2, Love = True, binary_love = binary_love, DA = False)


# print(accu)

#### DOMAIN ADAPTATION ####
# source_domain = ['urban']
# target_domain = ['rural']

# # Related to DS
# batch_size = 12
# transforms = None
# # get_LOVE_transforms()

# # Related to the network
# n_classes = 8
# bilinear = True
# starter_channels = 16
# up_layer = 4
# attention = True
# # resunet = False
# Love = True
# binary_love = False
# DA = True

# DS_args = [batch_size, transforms, DA, None, None]

# lr_d = 0.001
# lr_s = 0.001
# gamma = 3

# optim_args = [lr_s, lr_d]

# epochs = 70
# e_0 = 40
# l_max = 0.15

# DA_args = [epochs, e_0, l_max]


# ## ELIOTT BRION params
# # lr = 10**(-4)
# # optimizer = Adam
# # epochs = 150
# # e_0 = 50
# # lambda_max = [0.01, 0.03, 0.1, 0.3] [L3, L6, L9, L11]
                    
# network_args = [n_classes, bilinear, starter_channels, up_layer, attention, Love]

# seg_loss = FocalLoss(gamma = gamma, ignore_index = 0)
# domain_loss_function = torch.nn.BCEWithLogitsLoss()

# accu_function = JaccardIndex(task = 'multiclass', num_classes = n_classes, ignore_index = 0)

# best_model_accuracy, accu_target, best_overall, best_network, training_list = DANN_training_loop(source_domain, target_domain, DS_args, network_args, optim_args, DA_args, seg_loss = seg_loss, domain_loss = domain_loss_function, accu_function = accu_function, Love = Love)

# # # #### PREDICTIONS ####

# # # # predict_LoveDA('BestDANNModel.pt', ['urban', 'rural'])
# # # # predict_LoveDA('BestUrbanLoveDAModel.pt', ['rural'])

# # tr, val, test = get_LOVE_DataLoaders(['urban'], 2, transforms)

# # plot_GTvsPred_sample_images('BestDANNModel.pt', tr, 2, Love = True, binary_love = binary_love)

# # tr, val, test = get_LOVE_DataLoaders(['rural'], 2, transforms)

# # plot_GTvsPred_sample_images('BestDANNModel.pt', tr, 2, Love = True, binary_love= binary_love)


# plot_grad_flow(best_network.named_parameters())
