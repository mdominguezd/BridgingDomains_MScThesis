import torch

##### RELATED TO CUDA #####
def get_training_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


##### RELATED TO LOVEDA #####

def LOVE_resample_fly(i):
    if len(i.shape) == 4:
        i = i[:, :, ::4, ::4]
    elif len(i.shape) == 3:
        i = i[:, ::4, ::4]
    
    if i.unique().shape[0] > 8: # Hard code to avoid the transform to be done to the GT
        i = i/255

    return i