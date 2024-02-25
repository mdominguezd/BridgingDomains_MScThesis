import subprocess
import os
import time
import pandas as pd
from torchmetrics.classification import BinaryF1Score, JaccardIndex

from Dataset.ReadyToTrain_DS import Img_Dataset, get_DataLoaders, get_LOVE_DataLoaders
from Dataset.Transforms import get_transforms
from Dataset import Unzip_DS
from Training.Train_DomainOnly import *
from Training.Train_DANN import *
from Models.U_Net import UNet

from Models.Loss_Functions import FocalLoss

def HP_Tuning(dir, BS, LR, STCh, MU, Bi, gamma, VI, decay, atts, res, tr_size = 0.15, val_size = 0.75):
    """
        Function to perform Hyperparameter tuning for the networks to be trained.

        Input:
            - dir: Directory with the dataset to be used.
            - BS: List with values of batch_size to be considered during HP tuning.
            - LR: List with values of learning rate to be considered during HP tuning.
            - STCh: List with values of starting number of channels to be considered during HP tuning.
            - mu: List with values of momentum to be considered during HP tuning.
            - Bi: List with values of bilinear to be considered during HP tuning. (Only True or False possible)
            - gamma: List with values of gamma vlaues for the focal loss to be considered during HP tuning.
            - VI: List with values of vegetation indices (True or False)
            - decay: decay rate of learning rate.
            - atts: Inclusion or not of Attention gates.
            - res: Inclusion or not of residual connections on convolutional blocks.
        Output:
            - HP_values: (pandas.DataFrame) Dataframe with the results of each iteration of the hyperparameter tuning.
    """

    transforms = get_transforms()
    normalization = 'Linear_1_99'
    epochs = 20

    rows = []

    for bs in BS:
        for lr in LR:
            for stch in STCh:
                for mu in MU:
                    for bi in Bi:
                        for g in gamma:
                            for vi in VI:
                                for de in decay:
                                    for at in atts:
                                        for re in res:
                                            train_loader, val_loader, test_loader = get_DataLoaders(dir, bs, transforms, normalization, vi, train_split_size = tr_size, val_split_size = val_size)
                                            n_channels = next(enumerate(train_loader))[1][0].shape[1] #get band number fomr actual data
                                            n_classes = 2
                
                                            loss_function = FocalLoss(gamma = g)
                
                                            # Define the network
                                            network = UNet(n_channels, n_classes,  bi, stch, up_layer = 4, attention = at, resunet = re)
            
                                            start = time.time()
                                            f1_val, network_trained, spearman, no_l = training_loop(network, train_loader, val_loader, lr, mu, epochs, loss_function, decay = de, plot = False)
                                            end = time.time()
            
                                            rows.append([bs, lr, stch, mu, bi, g, vi, de, at, re, f1_val, end-start, spearman, no_l])
            
                                            HP_values = pd.DataFrame(rows)
                                            HP_values.columns = ['BatchSize','LR', 'StartCh', 'Momentum', 'Bilinear', 'gamma', 'VI', 'decay', 'attention', 'resnet', 'ValF1Score', 'Training time', 'Training rho', 'No_L']
                                            HP_values.to_csv('TempHyperParamTuning_'+dir+'.csv')
    
    return HP_values

def LoveDA_HP_Tuning(domain, BS, LR, STCh, MU, Bi, gamma, decay, atts, res, tr_size = 0.15, val_size = 0.75):
    """
        Function to perform Hyperparameter tuning for the networks to be trained on LoveDA dataset.

        Input:
            - dir: Directory with the dataset to be used.
            - BS: List with values of batch_size to be considered during HP tuning.
            - LR: List with values of learning rate to be considered during HP tuning.
            - STCh: List with values of starting number of channels to be considered during HP tuning.
            - mu: List with values of momentum to be considered during HP tuning.
            - Bi: List with values of bilinear to be considered during HP tuning. (Only True or False possible)
            - gamma: List with values of gamma vlaues for the focal loss to be considered during HP tuning.
            - decay: decay of learning rate.
            - atts: Boolean indicating if attention gates are used or not.
            - res: Boolean indicating if residua connections on convolutional blocks are used or not.
            
        Output:
            - top_5: Set of HP with which the top 5 validation F1 Score where otained
    """

    transforms = get_transforms()
    # normalization = 'Linear_1_99'
    epochs = 15

    rows = []

    for bs in BS:
        for lr in LR:
            for stch in STCh:
                for mu in MU:
                    for bi in Bi:
                        for g in gamma:
                            for de in decay:
                                for at in atts:
                                    for re in res:
                                        train_loader, val_loader, test_loader = get_LOVE_DataLoaders(domain, bs, train_split_size = tr_size, val_split_size = val_size)
                                        n_channels = next(enumerate(train_loader))[1]['image'].shape[1] #get band number fomr actual data
                                        n_classes = 8
            
                                        loss_function = FocalLoss(gamma = g, ignore_index = 0)
            
                                        # Define the network
                                        network = UNet(n_channels, n_classes,  bi, stch, up_layer = 4, attention = at, resunet = re)
        
                                        start = time.time()
                                        f1_val, network_trained, spearman, no_l = training_loop(network, train_loader, val_loader, lr, mu, epochs, loss_function, decay = de, plot = False, accu_function=JaccardIndex(task = 'multiclass', num_classes = n_classes, ignore_index = 0) , Love = True)
                                        end = time.time()
        
                                        rows.append([bs, lr, stch, mu, bi, g, de, at, re, f1_val, end-start, spearman, no_l])
        
                                        HP_values = pd.DataFrame(rows)
                                        HP_values.columns = ['BatchSize','LR', 'StartCh', 'Momentum', 'Bilinear', 'gamma', 'decay', 'attention', 'resunet', 'ValF1Score', 'Training time', 'Training rho', 'No_L']
                                        HP_values.to_csv('TempHyperParamTuning_LOVE.csv')
    
    return HP_values

def DANN_HP_Tuning(source_domain, target_domain, BS = [8], LR_s = [0.001], LR_d = [None], STCh = [16], MU = [None], Bi = [True], gamma = [2], VI = [True], atts = [True], e_0s = [15], l_max = [0.3], up_layers = [4], tr_size = 0.15, val_size = 0.75):
    """
    
    """

    transforms = get_transforms()
    normalization = 'Linear_1_99'
    DA = True
    n_classes = 2

    epochs = 80

    rows = []
    
    for bs in BS:
        for lrs in LR_s:
            for lrd in LR_d:
                for st in STCh:
                    for mu in MU:
                        for bi in Bi:
                            for gam in gamma:
                                for vi in VI:
                                    for att in atts:
                                        for e_0 in e_0s:
                                            for l_mx in l_max:
                                                for up_layer in up_layers:
                                                    
                                                    DS_args = [bs, transforms, normalization, vi, DA, tr_size, val_size]
                                                    network_args = [n_classes, bi, st, up_layer, att]

                                                    if lrd == None:
                                                        if mu == None:
                                                            optim_args = [lrs]
                                                    else:
                                                        if mu == None:
                                                            optim_args = [lrs, lrd]
                                                        else:
                                                            optim_args = [lrs, lrd, mu]

                                                    DA_args = [epochs, e_0, l_mx]

                                                    loss_fun = FocalLoss(gamma = gam)
                                                    
                                                    best_model_acc, target_acc, best_seg_disc_acc, best_network, training_list = DANN_training_loop(source_domain, target_domain, DS_args, network_args, optim_args, DA_args, seg_loss = loss_fun)

                                                    rows.append([bs, lrs, lrd, st, mu, bi, gam, vi, att, e_0, l_mx, up_layer, best_model_acc, target_acc, best_seg_disc_acc])
                                                    
                                                    HP_values = pd.DataFrame(rows)
                                                    HP_values.columns = ['BatchSize','LR_s', 'LR_d', 'StartCh', 'Momentum', 'Bilinear', 'gamma', 'VI', 'attention', 'e_0', 'l_max', 'up_layer', 'ValF1Score', 'ValTargetAccuracy', 'Disc+Seg']
                                                    HP_values.to_csv('TempHyperParamTuning_DANN.csv')
                                                    

                                                    
def run_HP_Tuning():
    
    BS = [16]
    LR = [0.01, 0.1, 1]
    decay = [0.75]
    STCh = [16, 32]
    MU = [0, 0.2]
    Bi = [True]
    gamma = [2, 4]
    VI = [True]
    atts = [True]
    res = [False]
    
    for i in range(2):
        i+=1
        HP_Tuning('TanzaniaSplit'+str(i+1), BS, LR, STCh, MU, Bi, gamma, VI, decay, atts, res, tr_size = 0.1, val_size = 0.1)
        HP_Tuning('IvoryCoastSplit'+str(i+1), BS, LR, STCh, MU, Bi, gamma, VI, decay, atts, res, tr_size = 0.05, val_size = 0.1)




