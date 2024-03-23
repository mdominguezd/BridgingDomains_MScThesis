import numpy as np
import torch
import pandas as pd

from Dataset.Transforms import get_transforms
from Dataset.ReadyToTrain_DS import get_DataLoaders
from Training.Train_DANN import evaluate
from Models.Loss_Functions import FocalLoss
from Validate.figures_for_validation import plot_GTvsPred_sample_images



def get_semisupervision_metrics():
    data_loaders = get_DataLoaders('TanzaniaSplit1', 10, get_transforms(), 'Linear_1_99', True)
    
    perc_incl_target = np.arange(5,56, 5)
    
    metrics = []
    
    for perc in perc_incl_target:
        
        net_fn = 'BestDANNModel_SemiSup_'+str(perc)+'.pt'
        net = torch.load(net_fn)

        splits = ['Train', 'Validation', 'Test']
        
        k = 0
        
        for dl in data_loaders:

            if k == 0: #Do not evaluate on training set
                k+=1
            else:
                met = evaluate(net, dl, FocalLoss(gamma = 2))[0]
                
                metrics.append([splits[k], met, len(dl), perc])
    
                k+=1

    df = pd.DataFrame(metrics)
    df.columns = ['split', 'metric', 'size', 'Percentage']

    return df