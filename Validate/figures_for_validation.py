import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
import seaborn as sns
from tqdm import tqdm
from sklearn.manifold import TSNE

plt.style.use('bmh')

from torchmetrics.classification import BinaryF1Score, JaccardIndex

from Dataset.ReadyToTrain_DS import *
from utils import get_training_device, LOVE_resample_fly

def plot_3fold_accuracies(domain, Stats):
    """
        Function to create barplots of the validation and test accuracies resulting of a three fold cross validation for domain only models (No Domain Adapatation). 
    """
    fig = plt.figure(figsize = (7,6))
    plt.bar(['Validation', 'Test'], Stats[0], yerr = Stats[1], capsize = 10)
    plt.title('F1-Score accuracy for Cashew classification in ' + domain + '\n3-fold CV')
    plt.tight_layout()
    fig.savefig(domain + '_accuracy.png', dpi = 200)

def plot_HyperparameterTuning(domain, y_HPs, x_HPs, folds = 3, dann = False):
    """
        Function to plot the results of hyperparameter tuning done for each of the domains.

        Inputs:
            - domain: (str) Name of the domain.
            - y_HPs: (list)
    """

    if dann:
        df = pd.read_csv('TempHyperParamTuning_DANN.csv')
        pv1 = pd.pivot_table(df,'ValF1Score',y_HPs, x_HPs)

        fig, ax = plt.subplots(1,1, figsize = (6,6), layout='constrained')
        
        sns.heatmap(pv1, vmin = 0, vmax = 1, cmap = 'RdYlGn', annot = True, ax = ax)
        
        ax.set_title('Validation F1-score')

        fig.savefig('HP_DANN.png')

    else:
        for i in range(folds):
            i+=1
            if i == 1:
                df = pd.read_csv('TempHyperParamTuning_'+domain+'Split'+str(i)+'.csv')
            else:
                df += pd.read_csv('TempHyperParamTuning_'+domain+'Split'+str(i)+'.csv')
    
        df /= folds
    
        pv_1 = pd.pivot_table(df,'ValF1Score',y_HPs, x_HPs)
        pv_2 = pd.pivot_table(df,'Training rho',y_HPs, x_HPs)
    
        fig, ax = plt.subplots(1,2, figsize = (12,6), layout='constrained')
        
        sns.heatmap(pv_1, vmin = 0, vmax = 1, cmap = 'RdYlGn', annot = True, ax = ax[0])
        
        ax[0].set_title('Validation F1-score')
        
        sns.heatmap(pv_2, vmin = 0, vmax = 1, cmap = 'RdYlGn', annot = True, ax = ax[1])
        
        ax[1].set_title('Training spearman\ncorrelation')
            
        plt.suptitle('Hyper parameter tuning '+ domain)
    
        fig.savefig('HP_'+domain+'.png')


def plot_GTvsPred_sample_images(network_fn, data_loader, num_images = 1, Love = False, binary_love = False, DA = True, device = 'cpu'):
    """
        
    """

    network = torch.load(network_fn)
    
    if Love:
        i, data = next(enumerate(data_loader))
        
        img = LOVE_resample_fly(data['image'])
        msk = LOVE_resample_fly(data['mask'])
        
        if binary_love:
            msk = (msk == 6).long()

        max = 7

        ACCU = JaccardIndex(task = 'multiclass', num_classes = 8).to(device)

        cmap = 'jet'
    else:
        i, data = next(enumerate(data_loader))
        
        img = data[0]
        msk = data[1][:,0,:,:]

        max = 1

        ACCU = BinaryF1Score().to(device)
        cmap = 'Greens'

    if img.shape[-4] < num_images:
        raise ValueError("num_images must be less or equal to the batch_size of the data loader")
            
    fig, ax = plt.subplots(num_images,3, figsize = (15, 5*(num_images)))

    network.to(device)
    img = img.to(device)
    msk = msk.to(device)
    
    pred = network(img)

    if DA:
        domains = ['Source' if i>0 else 'Target' for i in pred[1]]
        pred = pred[0]
        
    else:
        domains = ['Image']*num_images
    
    accuracy = ACCU(msk, pred.max(1)[1])
    
    for i in range(num_images):
        pred_ = pred[i].max(0)[1].cpu().numpy()

        img_ = img[i].cpu().numpy()
        img_ /= np.max(img_)
        
        msk_ = msk[i].cpu()
        
        if num_images == 1: 
            
            ax[0].imshow(np.transpose(img_[:3], (1,2,0)))
            ax[0].set_title(domains[i])
            ax[1].imshow(msk_, cmap = cmap, vmin = 0, vmax = max)
            ax[1].set_title('GT')
            ax[2].imshow(pred_, cmap = cmap, vmin = 0, vmax = max)
            ax[2].set_title('Predictions')
            
        else:
            
            ax[i,0].imshow(np.transpose(img_[:3], (1,2,0)))
            ax[i,0].set_title(domains[i])
            ax[i,1].imshow(msk_, cmap = cmap, vmin = 0, vmax = max)
            ax[i,1].set_title('GT')
            ax[i,2].imshow(pred_, cmap = cmap, vmin = 0, vmax = max)
            ax[i,2].set_title('Predictions')

        
    plt.suptitle('Batch accuracy:\n'+str(round(accuracy.numpy(), 2)))
    plt.tight_layout()

### TAKEN FROM: https://gist.github.com/Flova/8bed128b41a74142a661883af9e51490
def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().item())
            max_grads.append(p.grad.abs().max().item())

    fig = plt.figure(figsize = (18,6))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.04) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])



