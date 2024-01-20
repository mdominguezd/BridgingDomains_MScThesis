import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from tqdm import tqdm
from sklearn.manifold import TSNE

from torchmetrics.classification import BinaryF1Score, JaccardIndex

from Dataset.ReadyToTrain_DS import *

def get_training_device(): # This could go to an utils file
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plot_3fold_accuracies(domain, Stats):
    """
        Function to create barplots of the validation and test accuracies resulting of a three fold cross validation for domain only models (No Domain Adapatation). 
    """
    fig = plt.figure(figsize = (7,6))
    plt.bar(['Validation', 'Test'], Stats[0], yerr = Stats[1], capsize = 10)
    plt.title('F1-Score accuracy for Cashew classification in ' + domain + '\n3-fold CV')
    plt.tight_layout()
    fig.savefig(domain + '_accuracy.png', dpi = 200)

def plot_HyperparameterTuning():
    """
        
    """
    for domain in ['Tanzania', 'IvoryCoast']:
        
        for i in range(3):
            if i == 0:
                df = pd.read_csv('TempHyperParamTuning_'+domain+'Split'+str(i+1)+'.csv')
            else:
                df += pd.read_csv('TempHyperParamTuning_'+domain+'Split'+str(i+1)+'.csv')
            
            df['Validation rho'] = df['Training rho']
            df['Training time'] = 3*(df['Training time']/np.max(df['Training time']))
            
            piv = pd.pivot_table(df/3, ['Validation rho', 'ValF1Score', 'Training time'], ['LR', 'decay'], ['Momentum', 'gamma'])
            
        fig = plt.figure(figsize = (21,7))
        
        sns.heatmap(piv, cmap = 'RdYlGn', vmin = 0, vmax = 1, annot = True)
        
        plt.title('Hyper parameter tuning '+domain)
    
        fig.savefig('HP_'+domain+'.png')

def plot_GTvsPred_sample_images(network_fn, data_loader, num_images = 1, Love = False, device = 'cuda'):
    """
        
    """

    network = torch.load(network_fn)
    
    if Love:
        i, data = next(enumerate(data_loader))
        
        img = data['image']
        msk = data['mask']

        max = 6

        ACCU = JaccardIndex(task = 'multiclass', num_classes = 8).to(device)
    else:
        i, data = next(enumerate(data_loader))
        
        img = data[0]
        msk = data[1][:,0,:,:]

        max = 1

        ACCU = BinaryF1Score().to(device)

    if img.shape[-4] < num_images:
        raise ValueError("num_images must be less or equal to the batch_size of the data loader")
            
    fig, ax = plt.subplots(num_images,3, figsize = (15, 5*(num_images)))

    network.to(device)
    img = img.to(device)
    msk = msk.to(device)
    
    pred = network(img)
    
    accuracy = ACCU(msk, pred.max(1)[1])
    
    for i in range(num_images):
        pred_ = pred[i].max(0)[1].cpu().numpy()

        img_ = img[i].cpu().numpy()
        img_ /= np.max(img_)
        
        msk_ = msk[i].cpu()
        
        if num_images == 1: 
            
            ax[0].imshow(np.transpose(img_[:3], (1,2,0)))
            ax[1].imshow(msk_, cmap = 'jet', vmin = 0, vmax = max)
            ax[2].imshow(pred_, cmap = 'jet', vmin = 0, vmax = max)
            
        else:
            
            ax[i,0].imshow(np.transpose(img_[:3], (1,2,0)))
            ax[i,1].imshow(msk_, cmap = 'jet', vmin = 0, vmax = max)
            ax[i,2].imshow(pred_, cmap = 'jet', vmin = 0, vmax = max)

        
    plt.suptitle('Batch\naccuracy:'+str(accuracy))
    plt.tight_layout()
        



