import os
import rioxarray
import random
import torch
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T
from skimage import io
import numpy as np
from torch.utils.data import random_split
from torchgeo.datasets import LoveDA

def calculate_percentiles(img_folder, samples = 800):
    """
        Function to calculate 0.01 and 0.99 percentiles of the bands of planet images. These values will be later used for normalizing the dataset.

        Inputs:
            - img_folder: The name of the folder with the images.
            - samples: The number of images to take to calculate these percentiles, for computing reasons not all images are considered.
        Output:
            - vals: The mean 1% and 99% quantiles for the images analysed.
    """
    imgs = [fn for fn in os.listdir(img_folder) if 'StudyArea' in fn]

    random.seed(8)
    img_sample = random.sample(imgs, samples)
    quantiles = np.zeros((2,4))
    
    for i in img_sample:
        quantiles += rioxarray.open_rasterio(img_folder + "/" + i).quantile((0.01, 0.99), dim = ('x','y')).values
    
    vals = quantiles/len(img_sample)
    
    return vals

def get_LOVE_DataLoaders(domain = ['urban', 'rural'], batch_size = 4, transforms = None, only_get_DS = False, train_split_size = None, val_split_size = None):
    """
        Function to get the loaders for LoveDA dataset.

        Inputs:
            - domain: List with the scene parameter for the LoveDa dataset. It can include 'rural' and/or 'urban'.
            - batch_size: Number of images per batch.
            - transforms: Image augmentations that will be considered.
            - train_split_size: Amount of images from training split that will be considered. (Float between 0 and 1)
            - val_split_size: Amount of images from validation split that will be considered. (Float between 0 and 1)
            - only_get_DS: Boolean for only getting datasets instead of dataloaders.

        Output:
            - train_loader: Training torch LoveDA data loader
            - val_loader: Validation torch LoveDA data loader
            - test_loader: Test torch LoveDA data loader
    """
    if transforms != None:
        train_DS = LoveDA('LoveDA', split = 'train', scene = domain, download = True, transforms = transforms)
    else:
        train_DS = LoveDA('LoveDA', split = 'train', scene = domain, download = True)
        
    test_DS = LoveDA('LoveDA', split = 'test', scene = domain, download = True, transforms = transforms)
    val_DS = LoveDA('LoveDA', split = 'val', scene = domain, download = True, transforms = transforms)

    if train_split_size != None:
        if val_split_size == None:
            val_split_size = train_split_size
        train_DS, l = random_split(train_DS, [train_split_size, 1-train_split_size], generator=torch.Generator().manual_seed(8))
        val_DS, l = random_split(val_DS, [val_split_size, 1-val_split_size], generator=torch.Generator().manual_seed(8))
        test_DS, l = random_split(test_DS, [val_split_size, 1-val_split_size], generator=torch.Generator().manual_seed(8))
    
    train_loader = torch.utils.data.DataLoader(dataset=train_DS, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_DS, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_DS, batch_size=batch_size, shuffle=False)

    if only_get_DS:
        return train_DS, val_DS, test_DS
    else:
        return train_loader, val_loader, test_loader

def get_DataLoaders(dir, batch_size, transform, normalization, VI, only_get_DS = False, train_split_size = None, val_split_size = None, recalculate_perc = False):
    """
        Function to get the training, validation and test data loader for a specific dataset.

        Inputs:
            - dir: Directory with the name of the data to be used.
            - batch_size: Size of the batches used for training.
            - transform: torch composition of transforms used for image augmentation.
            - normaliztion: Type of normalization used.
            - VI: Boolean indicating if NDVI and NDWI are also used in training.
            - split_size: Float between 0 and 1 indicating the fraction of dataset to be used (Especifically useful for HP tuning)
            - only_get_DS: Boolean for only getting datasets instead of dataloaders.
            - train_split_size: (float) fraction of train split to be loaded. (number between 0 and 1)
            - val_split_size: (float) fraction of validation and test split to be loaded. (number between 0 and 1)

        Output:
            - train_loader: Training torch data loader
            - val_loader: Validation torch data loader
            - test_loader: Test torch data loader
    """
    
    train_DS = Img_Dataset(dir, transform, norm = normalization, VI=VI, recalculate_perc = recalculate_perc)
    val_DS = Img_Dataset(dir, split = 'Validation', norm = normalization, VI=VI, recalculate_perc = recalculate_perc)
    test_DS = Img_Dataset(dir, split = 'Test', norm = normalization, VI=VI, recalculate_perc = recalculate_perc)

    if train_split_size != None:
        if val_split_size == None:
            val_split_size = train_split_size
            
        train_DS, l = random_split(train_DS, [train_split_size, 1-train_split_size], generator=torch.Generator().manual_seed(8))
        val_DS, l = random_split(val_DS, [val_split_size, 1-val_split_size], generator=torch.Generator().manual_seed(8))
        test_DS, l = random_split(test_DS, [val_split_size, 1-val_split_size], generator=torch.Generator().manual_seed(8))
        
    train_loader = torch.utils.data.DataLoader(dataset=train_DS, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_DS, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_DS, batch_size=batch_size, shuffle=False)
    
    if only_get_DS:
        return train_DS, val_DS, test_DS
    else:
        return train_loader, val_loader, test_loader
    
# Default values calculated on cashew crops of both domains.
quant_CIV = np.array([[217.0,	528.0,	389.0,	2162.0],
                      [542.0,	896.0,	984.0,	3877.0]])

quant_TNZ = np.array([[209.0, 483.35, 335.0, 2560.0], 
                      [416.0, 723.65, 751.0, 3818.0]])

############################
###### DATASET CLASS #######
############################

class Img_Dataset(Dataset):
    """
        Class to manage the cashew dataset.
    """
    def __init__(self, img_folder, transform = None, split = 'Train', norm = 'Linear_1_99', VI = True, recalculate_perc = False):
        self.img_folder = img_folder
        self.transform = transform
        self.split = split
        self.norm = norm
        self.VI = VI

        # Depending of the domain the images will have different attributes (country and quantiles)
        if 'Tanzania'  in self.img_folder:
            self.country = 'Tanzania'
            
            if recalculate_perc:
                self.quant_TNZ = calculate_percentiles(img_folder)
            else:
                self.quant_TNZ = quant_TNZ
        else:
            self.country = 'IvoryCoast'
            
            if recalculate_perc:
                self.quant_CIV = calculate_percentiles(img_folder)
            else:
                self.quant_CIV = quant_CIV

    def __len__(self):
        """
            Method to calculate the number of images in the dataset.    
        """
        return sum([self.split in i for i in os.listdir(self.img_folder)])//2

    def plot_imgs(self, idx, VIs = False):
        """
            Method to plot a specific image of the dataset.
            
            Input:
                - self: The dataset class and its attributes.
                - idx: index of the image that will be plotted.
                - VIs: Boolean describing if vegetation indices should be plotted
        """

        im, g = self.__getitem__(idx)

        if VIs:
            fig, ax = plt.subplots(2,2,figsize = (12,12))

            ax[0,0].imshow(im[[2,1,0],:,:].permute(1,2,0))
            ax[0,0].set_title('Planet image')
            ax[0,1].imshow(g[0,:,:])
            ax[0,1].set_title('Cashew crops GT')

            VIs = im[4:6]

            g1=ax[1,0].imshow(VIs[0], cmap = plt.cm.get_cmap('RdYlGn', 5), vmin = -0.8, vmax = 0.8)
            ax[1,0].set_title('NDVI')
            fig.colorbar(g1)
            g2=ax[1,1].imshow(VIs[1], cmap = plt.cm.get_cmap('Blues_r', 5), vmin = -0.8, vmax = 0.8)
            ax[1,1].set_title('NDWI')
            fig.colorbar(g2)

        else:
            fig, ax = plt.subplots(1,2,figsize = (12,6))

            ax[0].imshow(im[[2,1,0],:,:].permute(1,2,0))
            ax[0].set_title('Planet image')
            ax[1].imshow(g[0,:,:])
            ax[1].set_title('Cashew crops GT')

        return fig


    def __getitem__(self, idx):
        """
            Method to get the tensors (image and ground truth) for a specific image.
        """
    
        conversion = T.ToTensor()

        img = io.imread(fname = self.img_folder + '/Cropped' + self.country + self.split + 'StudyArea_{:05d}'.format(idx) + '.tif').astype(np.float32)

        if self.VI:
            ndvi = (img[:,:,3] - img[:,:,2])/(img[:,:,3] + img[:,:,2]) 
            ndwi = (img[:,:,1] - img[:,:,3])/(img[:,:,3] + img[:,:,1])
            

        if self.norm == 'Linear_1_99':
            for i in range(img.shape[-1]):
                if 'Tanz' in self.img_folder:
                    img[:,:,i] = (img[:,:,i] - self.quant_TNZ[0,i])/(self.quant_TNZ[1,i] - self.quant_TNZ[0,i])
                elif 'Ivor' in self.img_folder:
                    img[:,:,i] = (img[:,:,i] - self.quant_CIV[0,i])/(self.quant_CIV[1,i] - self.quant_CIV[0,i])
        else:
            for i in range(img.shape[-1]):
                quant_CIV = np.array([[217.0,	528.0,	389.0,	2162.0],
                                      [542.0,	896.0,	984.0,	3877.0]])
                img[:,:,i] = (img[:,:,i] - quant_CIV[0,i])/(quant_CIV[1,i] - quant_CIV[0,i])

        if self.VI:
            ndvi = np.expand_dims(ndvi, axis = 2)
            ndwi = np.expand_dims(ndwi, axis = 2)
            img = np.concatenate((img, ndvi, ndwi), axis = 2)

        img = conversion(img).float()

        img = torchvision.tv_tensors.Image(img)

        GT = io.imread(fname = self.img_folder + '/Cropped' + self.country + self.split + 'GT_{:05d}'.format(idx) + '.tif').astype(np.float32)

        GT = torch.flip(conversion(GT), dims = (1,))

        GT = torchvision.tv_tensors.Image(GT)

        if self.transform != None:
            GT, img = self.transform(GT, img)

        return img, GT
    
    
    