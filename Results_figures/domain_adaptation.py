from Models.Loss_Functions import FocalLoss
from Dataset.ReadyToTrain_DS import get_LOVE_DataLoaders, get_DataLoaders
from Models.Loss_Functions import FocalLoss
from Training.Train_DomainOnly import evaluate
from Training.Train_DANN import evaluate as eval_da

import torch
import pandas as pd
from torchmetrics.classification import BinaryF1Score, JaccardIndex, MulticlassF1Score, BinaryConfusionMatrix

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

plt.style.use('bmh')
import warnings

from utils import get_training_device, LOVE_resample_fly

def get_3fold_accuracy(domain = 'CIV'):
    
    if (domain != 'CIV')|(domain != 'TNZ'):
        warnings.warn("domain must be CIV or TNZ")

    f1s = []
    f1s_tr = []
    
    x_f1s = []
    
    for i in range(3):
        net = torch.load(domain+'_Fold'+str(i+1)+'.pt')

        if domain == 'TNZ':
            tr, val, te = get_DataLoaders('TanzaniaSplit'+str(i+1), 1, None, 'Linear_1_99', True)
            x_tr, val, x_te = get_DataLoaders('IvoryCoastSplit'+str(i+1), 1, None, 'Linear_1_99', True)
        else:
            tr, val, te = get_DataLoaders('IvoryCoastSplit'+str(i+1), 1, None, 'Linear_1_99', True)
            x_tr, val, x_te = get_DataLoaders('TanzaniaSplit'+str(i+1), 1, None, 'Linear_1_99', True)

        f1s.append(evaluate(net, te, FocalLoss(2))[0])
        x_f1s.append(evaluate(net, x_te, FocalLoss(2))[0])
        f1s_tr.append(evaluate(net, tr, FocalLoss(2))[0])
        print('finished '+str(i+1)+'/3')
        
    return  f1s, np.mean(f1s), np.std(f1s), x_f1s, np.mean(x_f1s), np.std(x_f1s), f1s_tr, np.mean(f1s_tr), np.std(f1s_tr)

def get_prediction_comparison(dataset = 'LOVE', device = get_training_device()):

    if dataset == 'LOVE':
        S_net = torch.load('BestModel_Urban_70.pt').to(device)
        T_net = torch.load('BestModel_Rural_70_LR0.1.pt').to(device)
        DA_net = torch.load('BestDANNModel_LOVE.pt').to(device)

        tr, val, te = get_LOVE_DataLoaders(['rural'], 1)

        i, data = next(enumerate(val))

        img = LOVE_resample_fly(data['image']).to(device)
        mask = LOVE_resample_fly(data['mask']).to(device).cpu().numpy()

        max = 7
        cmap = 'jet'

        img_ = img[0].cpu().numpy()
        img_ /= np.max(img_)
        img_ = np.transpose(img_[:3], (1,2,0))

    elif dataset == 'Cashew':
        
        S_net = torch.load('CIV_Fold1.pt').to(device)
        T_net = torch.load('TNZ_Fold1.pt').to(device)
        DA_net = torch.load('BestDANNModel_Cashew.pt').to(device)

        tr, val, te = get_DataLoaders('TanzaniaSplit1', 1, None, 'Linear_1_99', True)

        i, data = next(enumerate(tr))

        img = data[0].to(device)
        mask = data[1][:,0,:,:].to(device).cpu().numpy()

        max = 1
        cmap = 'Greens'
        
        img_ = img[0].cpu().numpy()
        img_ = np.transpose(img_[:3], (1,2,0))

    else:
        warnings.warn("dataset argument must be LOVE or Cashew")


    pred_S = S_net(img)[0].max(0)[1].cpu().numpy()
    pred_T = T_net(img)[0].max(0)[1].cpu().numpy()
    pred_DA = DA_net(img)[0][0].max(0)[1].cpu().numpy()
    
    

    fig = plt.figure(figsize = (10,4), tight_layout=True)
    gs = GridSpec(2,5)

    ax = fig.add_subplot(gs[:, :2])
    ax.imshow(img_)
    ax.set_title('Image')

    ax_gt = fig.add_subplot(gs[0, 2])
    ax_gt.imshow(mask[0], vmin = 0, vmax = max, cmap = cmap)
    ax_gt.set_title('GT')

    ax_pS = fig.add_subplot(gs[0, 3])
    ax_pS.imshow(pred_S, vmin = 0, vmax = max, cmap = cmap)
    ax_pS.set_title('Source-only')

    ax_pT = fig.add_subplot(gs[1, 2])
    ax_pT.imshow(pred_T, vmin = 0, vmax = max, cmap = cmap)
    ax_pT.set_title('Target-only')

    ax_pD = fig.add_subplot(gs[1, 3])
    ax_pD.imshow(pred_DA, vmin = 0, vmax = max, cmap = cmap)
    ax_pD.set_title('DANN')

    if dataset == 'Cashew':
        legend_elements = [Patch(facecolor='white', edgecolor='k', label='Background'),
                           Patch(facecolor='darkgreen', edgecolor='k', label='Cashew')]
    else:
        labels = ['Background', 'Building', 'Road', 'Water', 'Barren', 'Forest', 'Agriculture']
        legend_elements = []
        cmap = matplotlib.cm.get_cmap('jet')
        
        for i in range(7):
            legend_elements.append(Patch(facecolor=cmap((i+1)/7), edgecolor='k', label=labels[i]))            

    ax_leg = fig.add_subplot(gs[:, 4])
    ax_leg.axis('off')
    ax_leg.legend(handles = legend_elements, loc = 'center left')
    
    return fig


def get_DANN_performance(Love = True):
    """
        Function to get a dataframe with the resulting performance of the lower and upper bound and the DANN model
    """
    if Love:
        T_net = torch.load('BestModel_Rural_70_LR0.1.pt')
        S_net = torch.load('BestModel_Urban_70.pt') # 
        DA_net = torch.load('BestDANNModel_LOVE.pt')
    else:
        S_net = torch.load('CIV_Fold1.pt')
        T_net = torch.load('TNZ_Fold1.pt')
        DA_net = torch.load('BestDANNModel_Cashew.pt')
    
    Low_score = 0 
    DA_score = 0
    
    if Love:
        S_tr, S_v, S_te = get_LOVE_DataLoaders(['urban'], 16)
        T_tr, T_v, T_te = get_LOVE_DataLoaders(['rural'], 16)

        accu = MulticlassF1Score(8, ignore_index = 0)

        Low_score += evaluate(S_net, T_v, FocalLoss(2), accu, Love = Love)[0]
        DA_score += eval_da(DA_net, T_v, FocalLoss(2), accu, Love = Love)[0]
    else:
        S_tr, S_v, S_te = get_DataLoaders('IvoryCoastSplit1', 16, None, 'Linear_1_99', True)
        T_tr, T_v, T_te = get_DataLoaders('TanzaniaSplit1', 16, None, 'Linear_1_99', True)

        Low_score += evaluate(S_net, T_te, FocalLoss(2), Love = Love)[0]
        DA_score += eval_da(DA_net, T_te, FocalLoss(2), Love = Love)[0]
        accu = BinaryF1Score()
    
    # Low_score += evaluate(S_net, T_v, FocalLoss(2), accu, Love = Love)[0]*val_im
    # Low_score += evaluate(S_net, T_tr, FocalLoss(2), accu, Love = Love)[0]*tr_im
    
    # DA_score += eval_da(DA_net, T_v, FocalLoss(2), accu, Love = Love)[0]*val_im
    # DA_score += eval_da(DA_net, T_tr, FocalLoss(2), accu, Love = Love)[0]*tr_im
    
    if Love:
        scores = pd.DataFrame([Low_score, evaluate(T_net, T_v, FocalLoss(2), accu, Love)[0], DA_score, evaluate(S_net, S_v, FocalLoss(2), accu, Love)[0]], ['Lower-bound', 'Upper-bound', 'DANN', 'Source_on_source'])
        scores = pd.concat([scores,
                            pd.DataFrame([evaluate(S_net, T_tr, FocalLoss(2), accu, Love)[0], 
                               evaluate(T_net, T_tr, FocalLoss(2), accu, Love)[0], 
                               eval_da(DA_net, T_tr, FocalLoss(2), accu, Love)[0],
                               evaluate(S_net, S_tr, FocalLoss(2), accu, Love)[0]], ['Lower-bound', 'Upper-bound', 'DANN', 'Source_on_source'])], axis =1)
        scores.columns = ['Validation', 'Train']
    else:
        scores = pd.DataFrame([Low_score, evaluate(T_net, T_te, FocalLoss(2), accu)[0], DA_score, evaluate(S_net, S_te, FocalLoss(2), accu, Love)[0]], ['Lower-bound', 'Upper-bound', 'DANN', 'Source_on_source'])
        scores = pd.concat([scores,
                            pd.DataFrame([evaluate(S_net, T_tr, FocalLoss(2), accu)[0],
                               evaluate(T_net, T_tr, FocalLoss(2), accu)[0], 
                               eval_da(DA_net, T_tr, FocalLoss(2), accu)[0],
                               evaluate(S_net, S_tr, FocalLoss(2), accu, Love)[0]], ['Lower-bound', 'Upper-bound', 'DANN', 'Source_on_source'])], axis = 1)
        scores.columns = ['Test', 'Train']

    return scores

def get_semisupervision_metrics():
    data_loaders = get_DataLoaders('TanzaniaSplit1', 16, None, 'Linear_1_99', True)
    
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
                met = eval_da(net, dl, FocalLoss(gamma = 2))[0]
                
                metrics.append([splits[k], met, len(dl), perc])
    
                k+=1

    df = pd.DataFrame(metrics)
    df.columns = ['split', 'metric', 'size', 'Percentage']

    return df

def plot_semisupervision_sample():
    tr, val, te = get_DataLoaders('TanzaniaSplit1', 1, None, 'Linear_1_99', True)

    i, data = next(enumerate(te))

    img = data[0].to('cuda')
    gt = data[1][:,0,:,:]

    perc_incl_target = np.arange(5,56, 5)

    fig, ax = plt.subplots(3,4, figsize = (16,12))

    row = 0
    col = 0
    
    ax[row, col].imshow(gt[0], cmap = 'Greens')
    ax[row, col].set_title('GT')
    
    for perc in perc_incl_target:
        
        col+=1
        
        net_fn = 'BestDANNModel_SemiSup_'+str(perc)+'.pt'
        net = torch.load(net_fn)
        
        pred_DA = net(img)[0][0].max(0)[1].cpu().numpy()
    
        ax[row, col].imshow(pred_DA, cmap = 'Greens')
        ax[row, col].set_title(str(perc)+'%')

        if (col != 0) & (col/3 == col//3):
            row += 1
            col = -1

    plt.tight_layout()
    fig.savefig('Semi_supervision_segmentations.png', dpi = 200)

def plot_TNZ_CIV_samples(device = get_training_device()):

    colors = [(0, 0, 0, 1), (0, 0, 0, 0)]  # (R, G, B, Alpha)
    black_to_transparent_cmap = LinearSegmentedColormap.from_list("BlackToTransparent", colors, N=256)

    
    trI, val, te = get_DataLoaders('IvoryCoastSplit1', 4, None, 'Linear_1_99', True)
    trT, val, te = get_DataLoaders('TanzaniaSplit1', 4, None, 'Linear_1_99', True)

    i, dataI = next(enumerate(trI))
    i, dataT = next(enumerate(trT))

    fig, ax = plt.subplots(4,2,figsize = (6,12))

    cmap = 'gray'

    for i in range(4):
        img = dataI[0].to(device)
        mask = dataI[1][:,0,:,:].to(device).cpu().numpy()
        
        max = 1
        
        img_ = img[i].cpu().numpy()

        img_ = np.transpose(img_[:3], (1,2,0))
            
        ax[i, 0].imshow(img_)
        ax[i, 0].imshow(mask[i], cmap = black_to_transparent_cmap , alpha = 0.7)

        if i==0:
            ax[i,0].set_title('Ivory Coast')
            ax[i,1].set_title('Tanzania')

        img = dataT[0].to(device)
        mask = dataT[1][:,0,:,:].to(device).cpu().numpy()
        
        max = 1
        
        img_ = img[i].cpu().numpy()
        img_ = np.transpose(img_[:3], (1,2,0))
    
        
        ax[i, 1].imshow(img_)
        ax[i, 1].imshow(mask[i], cmap = black_to_transparent_cmap , alpha = 0.7)

    return fig

def plot_confusion_matrix(network_fn, Dataset, Love = False, DA = False):
    """
        Function to obtain the confusion matrices
    """

    device = get_training_device()

    net = torch.load(network_fn).to(device)

    bcm = BinaryConfusionMatrix()

    tr, val, te = get_DataLoaders(Dataset, 4, None, 'Linear_1_99', True)

    with torch.no_grad():
        cms = []
        # Iterate over validate loader to get mean accuracy and mean loss
        for i, Data in enumerate(te):
            
            # The inputs and GT are obtained differently depending of the Dataset (LoveDA or our own DS)
            if Love:
                inputs = LOVE_resample_fly(Data['image'])
                GTs = LOVE_resample_fly(Data['mask'])
            else:
                inputs = Data[0]
                GTs = Data[1]
        
            inputs = inputs.to(device)
            GTs = GTs.type(torch.long).squeeze().to(device)

            pred = net(inputs)
            
            if DA == True:
                pred = pred[0]
        
            bcm = bcm.to(device)
        
            if (pred.max(1)[1].shape != GTs.shape):
                GTs = GTs[None, :, :]

            loss = FocalLoss(2)(pred, GTs)/GTs.shape[0]
        
            cm = bcm(pred.max(1)[1], GTs)

            cms.append(cm.to('cpu').numpy())

        cm = np.sum(cms, axis = 0)/(np.sum(cms))
        ax = sns.heatmap(cm, annot = True, vmin = 0, vmax = 1, cmap = 'RdYlGn')
        ax.set_yticks([0.5,1.5],['Background', 'Cashew'])
        ax.set_ylabel('GT')
        ax.set_xticks([0.5,1.5],['Background', 'Cashew'])
        ax.set_xlabel('Predicted')

        ax.set_title('Confusion matrix for ' + network_fn + '\napplied on ' + Dataset)

        fig = plt.gcf()

        plt.tight_layout()

        fig.savefig('Conf_matrix.png', dpi = 150)

    return cm
