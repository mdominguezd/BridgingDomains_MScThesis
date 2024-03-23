import torch
from torch import nn
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.gridspec as gridspec
from torchgeo.datasets import LoveDA

from Dataset.ReadyToTrain_DS import *

from utils import get_training_device, LOVE_resample_fly


def get_features_extracted(source_domain, target_domain, DS_args, network = None, network_filename = '', Love = False, cos_int = False, euc_int = False):
    """
        
    """
    device = get_training_device()

    if network == None:
        network = torch.load(network_filename, map_location = device) 
    
    if Love:
        source_loaders = get_LOVE_DataLoaders(source_domain, *DS_args)
        target_loaders = get_LOVE_DataLoaders(target_domain, *DS_args)

    else:
        source_loaders = get_DataLoaders(source_domain, *DS_args)
        target_loaders = get_DataLoaders(target_domain, *DS_args)

    n_batches = min(len(source_loaders[0]), len(target_loaders[0])) 

    batches = enumerate(zip(source_loaders[0], target_loaders[0]))

    cos = 0

    if cos_int or euc_int:
        d = 'Calculating distance metric'
    else:
        d = 'Getting features extracted'

    num_imgs = 4
    
    for i in tqdm(range(n_batches), desc = d):

        k, (source, target) = next(batches)

        if Love:
            source_input = LOVE_resample_fly(source['image']).to(device)
            target_input = LOVE_resample_fly(target['image']).to(device)  
        else:
            source_input = source[0].to(device)
            target_input = target[0].to(device)
        
        max_batch_size = np.min([source_input.shape[0], target_input.shape[0]])

        s_features = network.FE(source_input)[:max_batch_size].flatten(start_dim = 1).cpu().detach().numpy()
        t_features = network.FE(target_input)[:max_batch_size].flatten(start_dim = 1).cpu().detach().numpy()

        if i == 0:
            s_imgs = source_input[:num_imgs]
            # s_feats = s_features[:num_imgs]
            t_imgs = target_input[:num_imgs]
            # t_feats = t_features[:num_imgs]

        if cos_int:
            cos += cosine_sim(s_features, t_features)
        elif euc_int:
            cos += euc_dist(s_features, t_features)
        else:
            if i == 0:
                source_F = np.array(s_features)
                target_F = np.array(t_features)
            else:
                source_F = np.append(source_F, s_features, axis = 0)
                target_F = np.append(target_F, t_features, axis = 0)
                

    cos /= n_batches
    
    if cos_int or euc_int:
        return cos
    else:
        return source_F, target_F, s_imgs, t_imgs

def tSNE_source_n_target(source_F, target_F):
    """
        Function to visualize the features extracted for source and target domains.
    """

    X = np.append(source_F[:, :200000], target_F[:, :200000], axis = 0)
    
    domains = ['Source'] * source_F.shape[0] + ['Target'] * target_F.shape[0]
    
    comps = 2

    tsne = TSNE(comps, random_state = 123, perplexity = 50)

    tsne_X = tsne.fit_transform(X)

    fig, ax = plt.subplots(1,1,figsize = (7,4.5))

    sns.scatterplot(x = tsne_X[:,0], y = tsne_X[:,1], hue = domains, ax = ax, palette = ['darkblue', 'darkred'])

    plt.tight_layout()

    fig.savefig('t_SNE_simple.png', dpi = 200)
    
    for i in range(4):
        ax.scatter(x = tsne_X[i,0], y = tsne_X[i,1], s = 250, c = 'blue', zorder = -1)
        ax.text(x = tsne_X[i,0], y = tsne_X[i,1], s = str(i+1), color = 'white')
        ax.scatter(x = tsne_X[source_F.shape[0] + i,0], y = tsne_X[source_F.shape[0] + i,1], s = 250, c = 'red', zorder = -1)
        ax.text(x = tsne_X[source_F.shape[0] + i,0], y = tsne_X[source_F.shape[0]+ i,1], s = str(i+1), color = 'white')

    plt.tight_layout()

    fig.savefig('t_SNE.png', dpi = 200)

    return tsne_X

def cosine_sim(source_F, target_F):
    """
        Function to calculate cosine simmilarity between features extracted in source and target domain.
    """

    COS = nn.CosineSimilarity(dim = 1, eps = 1e-6)
    
    cos = COS(torch.tensor(source_F), torch.tensor(target_F)).mean()

    return cos.numpy()

def euc_dist(source_F, target_F):
    """
        Function to calculate de euclidean distance between vector of features extracted from source and target domain.
    """
    
    source_F = torch.tensor(source_F)
    target_F = torch.tensor(target_F)
    
    euc_dist = torch.cdist(source_F, target_F).mean()

    return euc_dist

def plot_img_n_features(network_fn, dir, index, Love = False):
    """
        Function to plot a specific image of the dataset with the features extracted by the network.
    """
    device = get_training_device()
    
    network = torch.load(network_fn, map_location = device) 

    if Love:
        img = LoveDA('LoveDA', split = 'train', scene = dir, download = True).__getitem__(index)['image']
        img = LOVE_resample_fly(img)[None, :, :, :].to(device)
    else:
        img = Img_Dataset(dir).__getitem__(index)[0][None,:,:,:].to(device)

    features = network.FE(img)

    img_ = img[0][:3].detach().cpu().numpy()
    img_ = np.transpose(img_[:3], (1,2,0))

    # Img_Dataset(dir).plot_imgs(index, False)

    fig = plt.figure(figsize = (16,8))
    
    gs = gridspec.GridSpec(4,8)    
    
    ax0 = fig.add_subplot(gs[:,:4])

    ax0.imshow(img_)

    ax0.get_yaxis().set_ticks([])
    ax0.get_xaxis().set_ticks([])

    for i in range(16):
        
        if i//4 == i/4:
            k = 0
            
        ax = fig.add_subplot(gs[i//4,4 + k])

        k += 1

        ax.imshow(features[0][i].detach().cpu().numpy())

        ax.get_yaxis().set_ticks([])
        ax.get_xaxis().set_ticks([])

    plt.tight_layout()
    plt.suptitle('Activation maps extracted with ' + network_fn, y = 1.02, fontsize = 24)
    fig.savefig('Act_maps.png', dpi = 200)
    