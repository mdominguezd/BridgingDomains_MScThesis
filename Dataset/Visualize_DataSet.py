import matplotlib.pyplot as plt
import numpy as np

def plot_LoveDA_sample(data_loader, num_images, save = False):
    """
        Function to plot sample of LoveDA dataset.

        Inputs:
            - data_loader: Loader with dataset that will be used to plot sample images. 
            - num_images: Number of images that will be plotted. Should be less than the batch size of the data loader.
            - save: Boolean to indicate if figure should be saved or not.
        Output:
            - None.
    """

    i, data = next(enumerate(data_loader))

    if data['image'].shape[-4] < num_images:
        raise ValueError("num_images must be less than the batch_size of the data loader")
        
    else: 
        
        fig, ax = plt.subplots(num_images,2, figsize = (10, 5*(num_images)))
        
        for i in range(num_images):
            img = data['image'][i].numpy()
            img /= np.max(img)
            msk = data['mask'][i]
            
            if num_images == 1: 
                ax[0].imshow(np.transpose(img, (1,2,0)))
                ax[1].imshow(msk, cmap = 'jet', vmin = 0, vmax = 6)
            else:
                ax[i,0].imshow(np.transpose(img, (1,2,0)))
                ax[i,1].imshow(msk, cmap = 'jet', vmin = 0, vmax = 6)

        plt.tight_layout()

        if save:
            fig.savefig("LoveDA_sampleDS.png", dpi = 200)