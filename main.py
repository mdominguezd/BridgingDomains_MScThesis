import subprocess
import os
import time
from torchmetrics.classification import BinaryF1Score, JaccardIndex
import pandas as pd

from Dataset.ReadyToTrain_DS import *
from Dataset.Transforms import *
from Dataset.Visualize_DataSet import *
from Dataset import Unzip_DS
from Training.Train_DomainOnly import *
from Training.Train_DANN import *
from Training.Hyper_Parameter_Tuning import *
from Validate.figures_for_validation import *
from Validate.predict import *
from Validate.domain_adaptation_performance import *

from Models.Loss_Functions import FocalLoss

####################################################################
######################## For Own Dataset ###########################
####################################################################

# DOMAIN ONLY CASE
run_DomainOnly(domain = 'Tanzania')
run_DomainOnly(domain = 'IvoryCoast')

# DOMAIN ADAPTED TRAINING
train_full_DANN()
