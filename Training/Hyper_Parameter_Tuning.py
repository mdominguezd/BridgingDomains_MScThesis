import subprocess
import os
import time
import pandas as pd
from torchmetrics.classification import BinaryF1Score

from Dataset.ReadyToTrain_DS import Img_Dataset
from Dataset.ReadyToTrain_DS import get_DataLoaders
from Dataset.Transforms import get_transforms
from Dataset import Unzip_DS
from Training.Train_DomainOnly import *
from Models.U_Net import UNet

from Models.Loss_Functions import FocalLoss

def HP_Tuning(dir, BS, LR, STCh, MU, Bi, gamma, VI):
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
        Output:
            - top_5: Set of HP with which the top 5 validation F1 Score where otained
    """

    transforms = get_transforms()
    normalization = 'Linear_1_99'
    epochs = 15

    rows = []

    for bs in BS:
        for lr in LR:
            for stch in STCh:
                for mu in MU:
                    for bi in BI:
                        for g in gamma:
                            for vi in VI:
                                train_loader, val_loader, test_loader = get_DataLoaders(dir, bs, transforms, normalization, vi, split_size = 0.1)
                                n_channels = next(enumerate(train_loader))[1][0].shape[1] #get band number fomr actual data
                                n_classes = 2
    
                                loss_function = FocalLoss(gamma = g)
    
                                # Define the network
                                network = UNet(n_channels, n_classes,  bi, stch, up_layer = 4)

                                start = time.time()
                                f1_val, network_trained = training_loop(network, train_loader, val_loader, lr, stch, mu, epochs, loss_function)
                                end = time.time()

                                rows.append([bs, lr, stch, mu, bi, g, vi, f1_val, end-start])

                                HP_values = pd.DataFrame(rows)
                                HP_values.columns = ['BatchSize','LR', 'StartCh', 'Momentum', 'Bilinear', 'gamma', 'VI', 'ValF1Score', 'Training time']
                                HP_values.to_csv('TempHyperParamTuning_'+dir+'.csv')

    top_5 = HP_values.sort_values('ValF1Score', ascending = False).head(5)
    
    return top_5