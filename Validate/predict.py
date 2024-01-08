import torch
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt
import os
import shutil
from tqdm import tqdm
import numpy as np

from Dataset.ReadyToTrain_DS import get_DataLoaders, get_LOVE_DataLoaders

def predict_LoveDA(network_filename, scene = ['rural', 'urban'], split = 'test', batch_size = 4):
    """
        Function to classify landcover using the LoveDA dataset and a network trained on that data.
    """
    if 'Predictions' not in os.listdir():
        os.mkdir('Predictions')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if 'rural' in scene:
        i = 4191
    else:
        i = 5167
        
    split_dict = {'test' : 2, 'val' : 1, 'train' : 0}
    
    loaders = get_LOVE_DataLoaders(domain = scene, batch_size = batch_size)
    
    loader = loaders[split_dict[split]]

    network = torch.load(network_filename)

    for img in tqdm(loader, 'Computing predictions'):

        img = img['image'].to(device)
        pred = (network(img).detach().max(1)[1] - 1)
        for b in range(img.size()[-4]):
            save_pred = pred[b].cpu().numpy().astype(np.uint8)
            im = Image.fromarray(save_pred)
            # .convert("L")
            im.save('Predictions/'+str(i)+'.png')
            # save_image(save_pred, 'Predictions/'+str(i)+'.png')
            i += 1

    shutil.make_archive('Result', 'zip', 'Predictions')

    