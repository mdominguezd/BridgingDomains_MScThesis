import numpy as np
import torch
import pandas as pd

from Dataset.Transforms import get_transforms
from Dataset.ReadyToTrain_DS import get_DataLoaders
from Training.Train_DANN import evaluate
from Models.Loss_Functions import FocalLoss
from Validate.figures_for_validation import plot_GTvsPred_sample_images



def get_metrics():
    tr, val, te = get_DataLoaders('TanzaniaSplit1', 10, get_transforms(), 'Linear_1_99', True)
    
    perc_incl_target = np.arange(5,21, 5)
    
    metrics = []
    
    for perc in perc_incl_target:
        
        net_fn = 'BestDANNModel_SemiSup_'+str(perc)+'.pt'
        net = torch.load(net_fn)
    
        valmet = evaluate(net, val, FocalLoss(gamma = 2))
        testmet = evaluate(net, te, FocalLoss(gamma = 2))
    
        metrics.append([valmet[0], testmet[0]])

    df = pd.DataFrame(metrics)
    df.columns = ['Validation', 'Test']
    df['Percentage'] = perc_incl_target

    return df