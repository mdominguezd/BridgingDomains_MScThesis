import torch
from torch import nn
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from Dataset.ReadyToTrain_DS import *

def get_training_device(): # This could go to an utils file
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_features_extracted(source_domain, target_domain, network_filename, DS_args, Love = False):
    """
        
    """
    device = get_training_device()
    network = torch.load(network_filename, map_location = device) 
    
    if Love:
        source_loaders = get_LOVE_DataLoaders(source_domain, *DS_args)
        target_loaders = get_LOVE_DataLoaders(target_domain, *DS_args)

    else:
        source_loaders = get_DataLoaders(source_domain, *DS_args)
        target_loaders = get_DataLoaders(target_domain, *DS_args)

    n_batches = min(len(source_loaders[0]), len(target_loaders[0]))

    batches = enumerate(zip(source_loaders[0], target_loaders[0]))
    
    for i in tqdm(range(n_batches), desc = 'Getting features extracted'):

        k, (source, target) = next(batches)

        if Love:
            source_input = source['image'].to(device)
            target_input = target['image'].to(device)  
        else:
            source_input = source[0].to(device)
            target_input = target[0].to(device)

        max_batch_size = np.min([source_input.shape[0], target_input.shape[0]])

        s_features = network.FE(source_input)[:max_batch_size]
        t_features = network.FE(target_input)[:max_batch_size]

        if i == 0:
            source_F = np.array(s_features.flatten(start_dim = 1).cpu().detach().numpy())
            target_F = np.array(t_features.flatten(start_dim = 1).cpu().detach().numpy())
        else:
            source_F = np.append(source_F, s_features.flatten(start_dim = 1).cpu().detach().numpy(), axis = 0)
            target_F = np.append(target_F, t_features.flatten(start_dim = 1).cpu().detach().numpy(), axis = 0)

    print(source_F.shape, target_F.shape)
    
    return source_F, target_F

def tSNE_source_n_target(source_F, target_F):
    """
        Function to visualize the features extracted for source and target domains.
    """

    X = np.append(source_F[:, :200000], target_F[:, :200000], axis = 0)
    
    domains = ['Source'] * source_F.shape[0] + ['Target'] * target_F.shape[0]
    
    comps = 2

    tsne = TSNE(comps, random_state = 123, perplexity = 50)

    tsne_X = tsne.fit_transform(X)

    fig, ax = plt.subplots(1,1,figsize = (9.5,5))

    sns.scatterplot(x = tsne_X[:,0], y = tsne_X[:,1], hue = domains, ax = ax, palette = ['darkblue', 'darkred'])

    ax.set_title('t-SNE of features extracted')

    fig.savefig('t_SNE.png')

    return tsne_X

def cosine_sim(source_F, target_F):
    """
        Function to calculate cosine simmilarity between features extracted in source and target domain.
    """

    COS = nn.CosineSimilarity(dim = 1, eps = 1e-6)
    
    cos = COS(torch.tensor(source_F), torch.tensor(target_F)).mean()

    return cos

def euc_dist(source_F, target_F):
    """
        Function to calculate de euclidean distance between vector of features extracted from source and target domain.
    """
    
    source_F = torch.tensor(source_F)
    target_F = torch.tensor(target_F)
    
    euc_dist = torch.cdist(source_F, target_F).mean()

    return euc_dist