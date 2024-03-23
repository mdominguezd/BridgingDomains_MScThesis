import torch
import matplotlib.pyplot as plt
from torchmetrics.classification import BinaryF1Score
from tqdm import tqdm
import pandas as pd

from Dataset.ReadyToTrain_DS import *
from Dataset.Transforms import *
from Models.U_Net import *
from Models.Loss_Functions import FocalLoss
from utils import get_training_device, LOVE_resample_fly
from Validate.domain_adaptation_performance import cosine_sim, get_features_extracted

plt.style.use('ggplot')

def initialize_Unet_DANN(n_channels, n_classes, bilinear = True, starter = 16, up_layer = 4, attention = True, Love = False, grad_rev_w = 1):
    """
        Function to initialize U-Net and the discriminator that will be trained using UNet-DANN

        Inputs:
            - n_channels: Number of channels of input images.
            - n_classes: Number of classes to be segmented on the images.
            - bilinear: Boolean used for upsamplimg method. (True: Bilinear is used. False: Transpose convolution is used.) [Default = True]
            - starter: Start number of channels of the UNet. [Default = 16]
            - up_layer: Upward step layer in which the U_Net is divided into Feature extractor and Classifier. [Default = 4]
            - attention: Boolean that describes if attention gates in the UNet will be used or not. [Default = True]

        Outputs:
            - network: U-Net architecture to be trained.
            - discriminator: Discriminator that will be trained.
    """
    device = get_training_device()

    # Calculate the number  of features that go in the fully connected layers of the discriminator
    if Love:
        img_size = 256
    else:
        img_size = 256
        
    in_feat = (img_size//(2**4))**2 * starter*(2**3) 
    
    seed = 123
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    network = UNetDANN(n_channels=n_channels, n_classes=n_classes, bilinear = bilinear, starter = starter, up_layer = up_layer, attention = attention, DA = True, in_feat = in_feat, grad_rev_w = grad_rev_w).to(device)

    return network

def evaluate(net, validate_loader, loss_function, accu_function = BinaryF1Score(), Love = False, binary_love = False, revgrad = 1):
    """
        Function to evaluate the performance of a network on a validation data loader.

        Inputs:
            - net: Pytorch network that will be evaluated.
            - validate_loader: Validation (or Test) dataset with which the network will be evaluated.
            - loss_function: Loss function used to evaluate the network.
            - accu_function: Accuracy function used to evaluate the network.

        Output:
            - metric: List with loss and accuracy values calculated for the validation/test dataset.
    """
    
    net.eval()  # Set the model to evaluation mode
    device = get_training_device()

    f1_scores = []
    losses = []

    with torch.no_grad():
        # Iterate over validate loader to get mean accuracy and mean loss
        for i, Data in enumerate(validate_loader):
            
            # The inputs and GT are obtained differently depending of the Dataset (LoveDA or our own DS)
            if Love:
                inputs = LOVE_resample_fly(Data['image'])
                GTs = LOVE_resample_fly(Data['mask'])
                if binary_love:
                    GTs = (GTs == 6).long()
                    
            else:
                inputs = Data[0]
                GTs = Data[1]
        
            inputs = inputs.to(device)
            GTs = GTs.type(torch.long).squeeze().to(device)
            pred = net(inputs, revgrad)[0]
        
            f1 = accu_function.to(device)
        
            if (pred.max(1)[1].shape != GTs.shape):
                GTs = GTs[None, :, :]

            loss = loss_function(pred, GTs)/GTs.shape[0]
            
            f1_score = f1(pred.max(1)[1], GTs)
            
            f1_scores.append(f1_score.to('cpu').numpy())
            losses.append(loss.to('cpu').numpy())

        metric = [np.nanmean(f1_scores), np.nanmean(losses)]   
        
    return metric

def evaluate_disc(network, validate_loaders, device, Love = False):
    """
        Function to evaluate the performance of the discriminator by the maximization of cosine simillarity

        Inputs:
            - network: Discriminator head used for the adversarial training
            - validate_loaders: 
            
    """

    network.eval()

    k = -1

    OA_final = []
    
    for loader in validate_loaders:
        
        k+=1
        
        OA = 0
        
        with torch.no_grad():
            # Iterate over validate loader to get mean accuracy and mean loss
            for i, Data in enumerate(loader):
                # The inputs and GT are obtained differently depending of the Dataset (LoveDA or our own DS)
                if Love:
                    inputs = LOVE_resample_fly(Data['image'])
                else:
                    inputs = Data[0]
    
                inputs = inputs.to(device)
                GTs = k*torch.ones(inputs.shape[0]).detach().numpy()

                features = network.FE(inputs)
                dom_preds = network.D(features, 0)

                preds = (dom_preds.detach().cpu().numpy() > 0)
                
                oa = (preds == GTs)

                OA += np.mean(oa)

        OA_final.append(OA / len(loader))
    
    OA_metric = np.mean(OA_final)
    
    return OA_metric
    

def DANN_training_loop(source_domain, target_domain, DS_args, network_args, optim_args, DA_args, Love = False, binary_love = False, seg_loss = torch.nn.CrossEntropyLoss(), domain_loss = torch.nn.BCEWithLogitsLoss(), accu_function = BinaryF1Score(), semi = False, semi_perc = 0.1):
    """
        Function to carry out the training of UNet-DANN.

        Inputs:
            - source_domain: Either str with name of the source domain ('IvoryCoast' or 'Tanzania') for own dataset or list with a str that has the name of the domain (['rural'] or ['urban']) for LoveDA dataset.
            - target_domain: Either str with name of the target domain ('IvoryCoast' or 'Tanzania') for own dataset or list with a str that has the name of the domain (['rural'] or ['urban']) for LoveDA dataset.
            - DS_args: List of arguments for dataset creation. 
                - For LoveDA: Should at least have: (batch_size, transforms, only_get_DS)
                - For own dataset: Should at least have: (batch_size, transform, normalization, VI, only_get_DS)
            - network_args: List of arguments for neural network creation. (e.g. n_classes, bilinear, starter, up_layer, attention)
    """

    device = get_training_device()

    if Love:
        if len(DS_args) < 3:
            raise Exception("The length of DS_args should be equal or greater than 3. Check the documentation for LoveDA.")
        source_DSs = get_LOVE_DataLoaders(source_domain, *DS_args)
        target_DSs = get_LOVE_DataLoaders(target_domain, *DS_args)
        n_channels = source_DSs[0].__getitem__(0)['image'].size()[-3]
        
    else:
        if len(DS_args) < 5:
            raise Exception("The length of DS_args should be equal or greater than 5. Check the documentation for own Dataset.")
        source_DSs = get_DataLoaders(source_domain, *DS_args)
        target_DSs = get_DataLoaders(target_domain, *DS_args)
        n_channels = source_DSs[0].__getitem__(0)[0].size()[-3]

    source_train_dataset = source_DSs[0]
    target_train_dataset = target_DSs[0]
    batch_size = DS_args[0]

    # Calculate number of batch iterations using both domains (The number of times the smallest dataset will need to be re-used for training.)
    source_n_batches = np.ceil(len(source_train_dataset)/(batch_size//2))
    target_n_batches = np.ceil(len(target_train_dataset)/(batch_size//2))
    
    n_batches = min(source_n_batches, target_n_batches)
    
    batch_iterations = np.ceil(max(source_n_batches, target_n_batches) / n_batches)

    # Create validation data loaders
    source_val_loader = torch.utils.data.DataLoader(dataset=source_DSs[1], batch_size=batch_size, shuffle=False)
    target_val_loader = torch.utils.data.DataLoader(dataset=target_DSs[1], batch_size=batch_size, shuffle=False)
    
    # Initialize the networks to be trained and the optimizer
    network = initialize_Unet_DANN(n_channels, *network_args)
    
    weight_decay = 1e-4
    
    if len(optim_args) == 3:
            optim = torch.optim.SGD([{'params': network.FE.parameters(), 'lr': optim_args[0]},
                                     {'params': network.C.parameters(), 'lr': optim_args[0]},
                                     {'params': network.D.parameters(), 'lr': optim_args[1]},
                                     ], weight_decay = weight_decay, momentum = optim_args[2])
    elif len(optim_args) > 1:
        optim = torch.optim.Adam([{'params': network.FE.parameters(), 'lr': optim_args[0]},
                                  {'params': network.C.parameters(), 'lr': optim_args[0]},
                                  {'params': network.D.parameters(), 'lr': optim_args[1]},
                                 ], weight_decay = weight_decay)
        
    else:
        optim = torch.optim.Adam([{'params': network.FE.parameters(), 'lr': optim_args[0]},
                                  {'params': network.C.parameters(), 'lr': optim_args[0]},
                                  {'params': network.D.parameters(), 'lr': optim_args[0]},
                                 ], weight_decay = weight_decay)
        
    # Create empty lists where segementation accuracy in source dataset and segmentation and domain loss will be stored.
    val_accuracy = []
    val_disc_accu = []
    val_accuracy_target = []
    segmen_loss_l = []
    train_accuracy_l = []
    train_disc_accuracy_l = []
    domain_loss_l = []

    eps = []

    source_loader = torch.utils.data.DataLoader(dataset=source_train_dataset, batch_size=batch_size//2, shuffle=True)
    target_loader = torch.utils.data.DataLoader(dataset=target_train_dataset, batch_size=batch_size//2, shuffle=True)

    if semi:
        sub = torch.utils.data.Subset(target_train_dataset, list(range(int(len(target_train_dataset)//(1/semi_perc)))))
        target_loader_sub = torch.utils.data.DataLoader(dataset=sub, batch_size=batch_size//2, shuffle=True)

    epochs, e_0, l_max = DA_args

    for epoch in tqdm(range(epochs), desc = 'Training UNet-DANN model'):

        batches = zip(source_loader, target_loader)
        n_batches = min(len(source_loader), len(target_loader))

        total_domain_loss = total_seg_accuracy = total_seg_loss = 0
        
        revgrad = np.max([0, l_max*(epoch - e_0)/(epochs - e_0)])

        for source, target in tqdm(batches, disable=True, total=n_batches):
            
            if Love:
                source_img = LOVE_resample_fly(source['image'])
                source_msk = LOVE_resample_fly(source['mask'])
                
                target_img = LOVE_resample_fly(target['image'])
                target_msk = LOVE_resample_fly(target['mask'])

            else:
                source_img = source[0]
                source_msk = source[1][:,0,:,:].to(torch.int64)
                
                target_img = target[0]
                target_msk = target[1]
                
                

            imgs = torch.cat([source_img, target_img])
            imgs = imgs.to(device)

            domain_gt = torch.cat([torch.ones(source_img.shape[0]),
                                   torch.zeros(target_img.shape[0])])
            
            domain_gt = domain_gt.to(device)
            mask_gt = source_msk.to(device)

            features = network.FE(imgs)
            dw = network.FE.DownSteps(imgs)
            
            seg_preds = network.C(features, dw)
            dom_preds = network.D(features, revgrad)
            
            # Calculate the loss function
            segmentation_loss = seg_loss(seg_preds[:source_img.shape[0]], mask_gt)
            discriminator_loss = domain_loss(dom_preds.squeeze(), domain_gt)

            if semi:
                
                target_sub = next(iter(target_loader_sub))
                    
                semitarget_img = target_sub[0].to(device)
                semitarget_gt = target_sub[1][:,0,:,:].to(torch.int64).to(device)
                
                features = network.FE(semitarget_img)
                dw = network.FE.DownSteps(semitarget_img)

                semi_preds = network.C(features, dw)

                seg_batch = seg_loss(semi_preds, semitarget_gt)
                
                segmentation_loss += seg_batch

            seg_imp = 1
            
            # Total loss
            loss = seg_imp*segmentation_loss + (2-seg_imp)*discriminator_loss

            # set the gradients of the model to 0 and perform the backward propagation
            optim.zero_grad()
            loss.backward()
            optim.step()

            total_domain_loss += discriminator_loss.item()
            total_seg_loss += segmentation_loss.item()
            accu_function = accu_function.to(device)
            accu = accu_function(seg_preds[:source_img.shape[0]].max(1)[1], mask_gt)
            total_seg_accuracy += accu.item()

        dom_loss = total_domain_loss / n_batches
        segmentation_loss = total_seg_loss / n_batches
        seg_accuracy = total_seg_accuracy / n_batches

        print('dom_loss, seg_loss, seg_accu', dom_loss, segmentation_loss, seg_accuracy)

        if (epoch//10 == epoch/10):
            #After 4 epochs, reduce the learning rate by a factor 
            optim.param_groups[0]['lr'] *= 0.75
            oa_val = 1
            # evaluate_disc(network, [source_loader, target_val_loader], device, Love)
        
        # Evaluate network on validation dataset
        f1_val, loss_val = evaluate(network, source_val_loader, seg_loss, accu_function, Love, binary_love, revgrad)
        val_accuracy.append(f1_val)
    
        f1_val_target, loss_val_target = evaluate(network, target_val_loader, seg_loss, accu_function, Love, binary_love, revgrad)
        val_accuracy_target.append(f1_val_target)
        
        #update values in DS_args for cosine similarity computation
        # DS_args[-3] = False
        # DS_args[-2] = 0.025
        # DS_args[-1] = 0.025
        
        

        eps.append(epoch + 1)
        val_disc_accu.append(oa_val)
        segmen_loss_l.append(segmentation_loss)
        train_accuracy_l.append(seg_accuracy)
        domain_loss_l.append(dom_loss)

        # Selection of best model so far using validation dataset.
        # Relative importance of segmentation over discrimination (0 to 5)
        rel_imp_seg = 3

        overall = ((5-rel_imp_seg)*(dom_loss) + rel_imp_seg*(f1_val))/5

        print('disc_accu, f1val, overall', oa_val, f1_val, overall)

        if epoch == 0:
            best_model_f1 = f1_val
            best_oa = oa_val
            best_overall = overall
            target_f1 = f1_val_target
            torch.save(network, 'BestDANNModel.pt')
            best_network = network
        else:
            if best_overall < overall:
                best_model_f1 = f1_val
                best_oa = oa_val
                best_overall = overall
                print(best_overall)
                target_f1 = f1_val_target
                torch.save(network, 'BestDANNModel.pt')
                best_network = network

        if epoch == e_0:
            best_DA_f1 = f1_val
            torch.save(network, 'BestDANNModelAfter_e0.pt')
            best_network = network
        elif epoch > e_0:
            if best_DA_f1 < f1_val:
                best_DA_f1 = f1_val
                torch.save(network, 'BestDANNModelAfter_e0.pt')
                best_network = network
                
        fig = plt.figure(figsize = (7,5))
        
        plt.plot(eps, segmen_loss_l, '-k', label = 'Segmentation loss')
        plt.plot(eps, domain_loss_l, '-r', label = 'Domain loss')
        plt.plot(eps, train_accuracy_l, '--g', label = 'Train segmentation accuracy')
        plt.axvline(x = e_0, color = 'darkred', label = 'e_0')

        plt.plot(eps, val_accuracy, label = 'Source domain val accuracy')
        plt.plot(eps, val_accuracy_target, label = 'Target domain val accuracy')
        # plt.plot(eps, val_disc_accu, label = 'Discrimination accuracy')
        # plt.plot(eps, train_disc_accuracy_l, '--y', label = 'Train discriminator accuracy')

        plt.ylim((0,1.1))
        plt.xlabel('Epoch')

        plt.legend()

        fig.savefig('DANN_Training.png', dpi = 100)
        plt.close()

    training_list = pd.DataFrame([eps, segmen_loss_l, domain_loss_l, train_accuracy_l, val_accuracy, val_accuracy_target, val_disc_accu])

    training_list.to_csv('Training_loop.csv')

    torch.save(network, 'LastDANNModel.pt')

    return best_model_f1, target_f1, best_overall, best_network, training_list


def train_full_DANN():
    
    DS_args = [8, get_transforms(), 'Linear_1_99', True, True, None, None]

    ## Related to the network
    n_classes = 2
    bilinear = True
    sts = 16
    up_layer = 4
    att = True
    
    network_args = [n_classes, bilinear, sts, up_layer, att]
    
    lr_s = 0.0001
    lr_d = 0.0001
    
    optim_args = [lr_s, lr_d]
    
    epochs = 80
    e_0 = 40
    l_max = 0.1
    
    DA_args = [epochs, e_0, l_max]
    
    best_model_f1, target_f1, best_overall, best_network, training_list = DANN_training_loop('IvoryCoastSplit1', 'TanzaniaSplit1', DS_args, network_args, optim_args, DA_args, seg_loss = FocalLoss(4))
    