import subprocess
import os
from Dataset.ReadyToTrain_DS import Img_Dataset
from Dataset.ReadyToTrain_DS import get_DataLoaders
from Dataset import Transforms
from Dataset import Unzip_DS
from Training import Train_DomainOnly 

# Once datasets have been downloaded (Using DS_Download.sh) you can unzip them
Unzip_DS.UnzipFolders("Tanzania")
Unzip_DS.UnzipFolders("IvoryCoast")

# Build Dataset class
train_loader, val_loader, test_loader = get_DataLoaders('TanzaniaSplit1')





