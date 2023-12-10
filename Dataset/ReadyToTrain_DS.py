
###### DATASET CLASS #######
# These values were calculated using the histograms on notebook 01_02_DataDistributionShift.ipynb
means_CIV = [338.995536,	677.268924,	630.248048,	2874.836857]
oneperc_CIV = [217.0,	528.0,	389.0,	2162.0]
ninenine_CIV = [542.0,	896.0,	984.0,	3877.0]
std_CIV = [67.369635,	79.766812,	131.564375,	365.127796]

means_TNZ = [299.61809,	584.028463,	545.825534,	3020.286079]
oneperc_TNZ = [209.0,	483.35,	335.0,	2560.0]
ninenine_TNZ = [	416.0,	723.65,	751.0	,3818.0]
std_TNZ = [63.56492,	72.122137,	103.400951,	295.013028]

class Img_Dataset(Dataset):
    def __init__(self, img_folder, transform = None, split = 'Train', norm = 'StandardScaler', VI = True):
        self.img_folder = img_folder
        self.transform = transform
        self.split = split
        self.norm = norm
        self.VI = VI

        if 'Tanzania'  in self.img_folder:
            self.country = 'Tanzania'
        else:
            self.country = 'IvoryCoast'

    def __len__(self):
        return sum([self.split in i for i in os.listdir(self.img_folder)])//2

    def plot_imgs(self, idx, VIs = False):

        im, g = self.__getitem__(idx)

        if VIs:
            fig, ax = plt.subplots(2,2,figsize = (12,12))

            ax[0,0].imshow(im[[2,1,0],:,:].permute(1,2,0))
            ax[0,0].set_title('Planet image')
            ax[0,1].imshow(g[0,:,:])
            ax[0,1].set_title('Cashew crops GT')

            VIs = im[4:6]

            g1=ax[1,0].imshow(VIs[0], cmap = plt.cm.get_cmap('RdYlGn', 5), vmin = 0, vmax = 1)
            ax[1,0].set_title('NDVI')
            fig.colorbar(g1)
            g2=ax[1,1].imshow(VIs[1], cmap = plt.cm.get_cmap('Blues_r', 5), vmin = 0, vmax = 1)
            ax[1,1].set_title('NDWI')
            fig.colorbar(g2)

        else:
            fig, ax = plt.subplots(1,2,figsize = (12,6))

            ax[0].imshow(im[[2,1,0],:,:].permute(1,2,0))
            ax[0].set_title('Planet image')
            ax[1].imshow(g[0,:,:])
            ax[1].set_title('Cashew crops GT')


    def __getitem__(self, idx):
        #__getitem__ asks for the sample number idx.

        conversion = T.ToTensor()

        img = io.imread(fname = self.img_folder + '/Cropped' + self.country + self.split + 'StudyArea_{:05d}'.format(idx) + '.tif').astype(np.float32)

        if self.VI:
            # Should I normalize this values between 0 and 1?
            if self.norm == 'Linear_1_99':
                ndvi = ((img[:,:,3] - img[:,:,2])/(img[:,:,3] + img[:,:,2]) - 0.37)/(0.86 - (0.37))
                ndwi = ((img[:,:,1] - img[:,:,3])/(img[:,:,3] + img[:,:,1]) - (-0.79))/((-0.41) - (-0.79))
            else:
                ndvi = (img[:,:,3] - img[:,:,2])/(img[:,:,3] + img[:,:,2])
                ndwi = (img[:,:,1] - img[:,:,3])/(img[:,:,3] + img[:,:,1])

        if self.norm == 'StandardScaler':
            for i in range(img.shape[-1]):
                if 'Tanz' in self.img_folder:
                    img[:,:,i] = (img[:,:,i] - means_TNZ[i])/(std_TNZ[i])
                elif 'Ivor' in self.img_folder:
                    img[:,:,i] = (img[:,:,i] - means_CIV[i])/(std_CIV[i])

        elif self.norm == 'Linear_1_99':
            for i in range(img.shape[-1]):
                if 'Tanz' in self.img_folder:
                    img[:,:,i] = (img[:,:,i] - oneperc_TNZ[i])/(ninenine_TNZ[i] - oneperc_TNZ[i])
                elif 'Ivor' in self.img_folder:
                    img[:,:,i] = (img[:,:,i] - oneperc_CIV[i])/(ninenine_CIV[i] - oneperc_CIV[i])

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