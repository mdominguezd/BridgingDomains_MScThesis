import numpy as np
import torchvision.transforms.v2 as T

# To ensure reproducible results.
seed = 8
np.random.seed(seed)

def brightness(i, increase = 0.1, prob = 0.5):
    """
        Function to augment images it increases or decreases the brightness of the image by a value of 0 to 20%
    """
    if i.unique().shape[0] != 2: # Hard code to avoid the transform to be done to the GT
        p = np.random.random(1)
        if p < prob:
            p_inc = np.random.random(1)
            i = i*(1 + increase*p_inc)
            i[i>1] = 1.0
        else:
            p_dec = np.random.random(1)
            i = i*((1 - increase*p_dec))

    return i.float()


### ALL transforms performed on the dataset:
def get_transforms():
    """
        Function that will return the transform to be made on the fly to data.
    """
    transform = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.Lambda(brightness)
    ])
    return transform