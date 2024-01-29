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

# # Once datasets have been downloaded (Using DS_Download.sh) you can unzip them
# Unzip_DS.UnzipFolders("Tanzania")
# Unzip_DS.UnzipFolders("IvoryCoast")

# If needed run Hyperparameter tuning to get the optimal HPs
# BS = [8]
# LR = [0.001, 0.01, 0.1, 1.5, 3]
# decay = [0.8]
# STCh = [16]
# MU = [0]
# Bi = [True]
# gamma = [1, 2, 4]
# VI = [True]
# atts = [True, False]
# res = [False, True]

# for i in range(1):
#     # HP_Tuning('TanzaniaSplit'+str(i+1), BS, LR, STCh, MU, Bi, gamma, VI, decay, atts, res, tr_size = 0.05, val_size = 0.1)
#     HP_Tuning('IvoryCoastSplit'+str(i+1), BS, LR, STCh, MU, Bi, gamma, VI, decay, atts, res, tr_size = 0.05, val_size = 0.1)
    
# Hyperparameters for DOMAIN ONLY TRAINING
# domain = 'Tanzania'

# ## Related to DS
# batch_size = 8
# transforms = get_transforms()
# normalization = 'Linear_1_99'
# VI = True
# DA = False

# ## Related to the network
# n_classes = 2
# bilinear = True
# starter_channels = 8
# up_layer = 4
# attention = True
# resunet = True

# ## Related to training and evaluation
# number_epochs = 20
# learning_rate = 0.1
# momentum = 0.0
# loss_function = FocalLoss(gamma = 1)
# accu_function = BinaryF1Score()
# device = get_training_device()

# DS_args = [batch_size, transforms, normalization, VI, DA, 1, 1]
# network_args = [n_classes, bilinear, starter_channels, up_layer, attention, resunet]
# training_args = [learning_rate, momentum, number_epochs, loss_function]
# eval_args = [loss_function, accu_function]

# Stats = train_3fold_DomainOnly(domain, DS_args, network_args, training_args, eval_args)

# plot_3fold_accuracies(domain, Stats)

# For Features extracted analysis
# S_, T_ = get_features_extracted('IvoryCoastSplit1', 'TanzaniaSplit1', 'OverallBestModel'+domain+'.pt', DS_two_doms, Love = False)

# _ = tSNE_source_n_target(S_, T_)

# cos = cosine_sim(S_, T_)
# euc = euc_dist(S_, T_)

# print('cosine simmilarity:', cos, '\neuclidean distance:', euc)

#### DOMAIN ADAPTATION ####
# source_domain = 'IvoryCoastSplit2'
# target_domain = 'TanzaniaSplit2'

# ## Related to DS
# batch_size = 16
# transforms = get_transforms()
# normalization = 'Linear_1_99'
# VI = True
# DA = True

# ## Related to the network
# n_classes = 2
# bilinear = True
# sts = [16]
# up_layer = 4
# atts = [True]
# resunet = False

# lr_s = [0.0001]
# lr_d = [0.0001]
# momentums = [0]
# gammas = [3]

# epochs = 70
# e_0 = 35
# l_max = 0.3
# Love = False
# binary_love = False

# domain_loss_function = torch.nn.BCEWithLogitsLoss(reduction = 'mean')
# accu_function = BinaryF1Score()

# rows = []

# for lr in lr_s:
#     for lrd in lr_d:
#         for momentum in momentums:
#             for gamma in gammas:
#                 for attention in atts:
#                     for starter_channels in sts:
#                         DS_args = [batch_size, transforms, normalization, VI, DA, 0.1, None]
#                         network_args = [n_classes, bilinear, starter_channels, up_layer, attention, resunet]
                    
#                         seg_loss_function = FocalLoss(gamma = gamma)
#                         best_model_accuracy, target, best_overall, best_network = DANN_training_loop(source_domain, target_domain, DS_args, network_args, lr, lrd, momentum, epochs, e_0, l_max, Love, binary_love, seg_loss_function, domain_loss_function, accu_function)
            
#                         rows.append([starter_channels, lr, lrd, momentum, gamma, attention, best_model_accuracy, target, best_overall])

#                         df = pd.DataFrame(rows)
#                         df.columns = ['StCh', 'LR_seg', 'LR_d', 'momentum', 'gamma', 'attention', 'F1-Source', 'F1-Target', 'best_overall']
#                         df.to_csv('HP_DANN.csv')

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
source_domain = ['urban']
target_domain = ['rural']

# Related to DS
batch_size = 12
transforms = None
# get_LOVE_transforms()

# Related to the network
n_classes = 8
bilinear = True
starter_channels = 16
up_layer = 4
attentions = [True]
# resunet = False
Love = True

binary_love = False

DS_args = [batch_size, transforms, True, 0.1, 0.1]


lrs_d = [0.001]
lrs_s = [0.001]
gammas = [3]
momentums =  [0.0]

epochs = 70
e_0 = 40
l_mxs = [0.15]

## ELIOTT BRION params
# lr = 10**(-4)
# optimizer = Adam
# epochs = 150
# e_0 = 50
# lambda_max = [0.01, 0.03, 0.1, 0.3] [L3, L6, L9, L11]

rows = []

for lr_disc in lrs_d:
    for lr_seg in lrs_s:
        for gamma in gammas:
            for momentum in momentums:
                for attention in attentions:
                    for l_max in l_mxs:
                    
                        network_args = [n_classes, bilinear, starter_channels, up_layer, attention, Love]
                    
                        seg_loss_function = FocalLoss(gamma = gamma, ignore_index = 0)
                        domain_loss_function = torch.nn.BCEWithLogitsLoss()
                        accu_function = JaccardIndex(task = 'multiclass', num_classes = n_classes, ignore_index = 0)
                        
                        best_model_accuracy, accu_target, best_overall, best_network, training_list = DANN_training_loop(source_domain, target_domain, DS_args, network_args, lr_seg, lr_disc, momentum, epochs, e_0, l_max, Love, binary_love, seg_loss_function, domain_loss_function, accu_function)
            
                        rows.append([lr_seg, lr_disc, gamma, momentum, attention, l_max, best_model_accuracy, accu_target, best_overall])
            
                        df = pd.DataFrame(rows)
                        df.columns = ['LR_seg', 'LR_disc', 'gamma','momentum', 'attention', 'l_max', 'IOU-Source', 'IOU-Target', 'IOU-Source + Discriminator']
                        df.to_csv('HP_LOVE_DANN_2.csv')

# # # #### PREDICTIONS ####

# # # # predict_LoveDA('BestDANNModel.pt', ['urban', 'rural'])
# # # # predict_LoveDA('BestUrbanLoveDAModel.pt', ['rural'])

# # tr, val, test = get_LOVE_DataLoaders(['urban'], 2, transforms)

# # plot_GTvsPred_sample_images('BestDANNModel.pt', tr, 2, Love = True, binary_love = binary_love)

# # tr, val, test = get_LOVE_DataLoaders(['rural'], 2, transforms)

# # plot_GTvsPred_sample_images('BestDANNModel.pt', tr, 2, Love = True, binary_love= binary_love)


# plot_grad_flow(best_network.named_parameters())
