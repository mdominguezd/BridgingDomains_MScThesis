import subprocess
import os
from Dataset.ReadyToTrain_DS import Img_Dataset
from Dataset import Transforms
from Dataset import Unzip_DS

# Once datasets have been downloaded (Using DS_Download.sh) you can unzip them
Unzip_DS.UnzipFolders("Tanzania")
Unzip_DS.UnzipFolders("IvoryCoast")







