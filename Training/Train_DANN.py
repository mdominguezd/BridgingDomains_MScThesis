import torch
import matplotlib.pyplot as plt
from torchmetrics.classification import BinaryF1Score
from tqdm import tqdm
import pandas as pd

from Dataset.ReadyToTrain_DS import *
from Models.U_Net import *
from utils import get_training_device, LOVE_resample_fly

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

def evaluate_disc(network, validate_loaders, device, Love = False, revgrad = 0):
    """
        Function to evaluate the performance of the discriminator

        Inputs:
            - discriminator: Discriminator head used for the adversarial training
            
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
                dom_preds = network.D(features, revgrad)

                preds = (dom_preds.detach().cpu().numpy() > 0)
                
                oa = (preds == GTs)

                OA += np.mean(oa)

        OA_final.append(OA / len(loader))
    
    OA_metric = np.mean(OA_final)
    
    return OA_metric
    

def DANN_training_loop(source_domain, target_domain, DS_args, network_args, learning_rate_seg, learning_rate_disc, momentum, epochs, e_0, l_max, Love = False, binary_love = False, seg_loss = torch.nn.CrossEntropyLoss(), domain_loss = torch.nn.BCEWithLogitsLoss(), accu_function = BinaryF1Score()):
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

    source_bigger_target = source_n_batches > target_n_batches
    
    batch_iterations = np.ceil(max(source_n_batches, target_n_batches) / n_batches)

    # Create validation data loaders
    source_val_loader = torch.utils.data.DataLoader(dataset=source_DSs[1], batch_size=batch_size, shuffle=False)
    target_val_loader = torch.utils.data.DataLoader(dataset=target_DSs[1], batch_size=batch_size, shuffle=False)
    
    # Initialize the networks to be trained and the optimizer
    network = initialize_Unet_DANN(n_channels, *network_args)
    optim = torch.optim.Adam([{'params': network.FE.parameters(), 'lr': learning_rate_seg},
                              {'params': network.C.parameters(), 'lr': learning_rate_seg},
                              {'params': network.D.parameters(), 'lr': learning_rate_disc},
                              ])
    # , momentum=momentum)
    
    # Create empty lists where segementation accuracy in source dataset and segmentation and domain loss will be stored.
    val_accuracy = []
    val_disc_accu = []
    val_accuracy_target = []
    segmen_loss_l = []
    train_accuracy_l = []
    train_disc_accuracy_l = []
    domain_loss_l = []

    eps = []

    source_loader = torch.utils.data.DataLoader(dataset=source_DSs[0], batch_size=batch_size//2, shuffle=True)
    target_loader = torch.utils.data.DataLoader(dataset=target_DSs[0], batch_size=batch_size//2, shuffle=True)

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
            # torch.save(network, 'DANNModel_epoch'+str(epoch)+'.pt')
        
        # Evaluate network on validation dataset
        f1_val, loss_val = evaluate(network, source_val_loader, seg_loss, accu_function, Love, binary_love, revgrad)
        val_accuracy.append(f1_val)
    
        f1_val_target, loss_val_target = evaluate(network, target_val_loader, seg_loss, accu_function, Love, binary_love, revgrad)
        val_accuracy_target.append(f1_val_target)

        oa_val = evaluate_disc(network, [source_val_loader, target_val_loader], device, Love, revgrad)

        eps.append(epoch + 1)
        val_disc_accu.append(oa_val)
        segmen_loss_l.append(segmentation_loss)
        train_accuracy_l.append(seg_accuracy)
        domain_loss_l.append(dom_loss)

        # Selection of best model so far using validation dataset.
        # Relative importance of segmentation over discrimination (0 to 5)
        rel_imp_seg = 4.75

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
                
        fig = plt.figure(figsize = (7,5))
        
        plt.plot(eps, segmen_loss_l, '-k', label = 'Segmentation loss')
        plt.plot(eps, domain_loss_l, '-r', label = 'Domain loss')
        plt.plot(eps, train_accuracy_l, '--g', label = 'Train segmentation accuracy')
        plt.axvline(x = e_0, color = 'darkred', label = 'e_0')

        plt.plot(eps, val_accuracy, label = 'Source domain val accuracy')
        plt.plot(eps, val_accuracy_target, label = 'Target domain val accuracy')
        plt.plot(eps, val_disc_accu, label = 'Discrimination accuracy')
        # plt.plot(eps, train_disc_accuracy_l, '--y', label = 'Train discriminator accuracy')

        plt.ylim((0,1.1))

        plt.legend()

        fig.savefig('DANN_Training.png', dpi = 100)
        plt.close()

    training_list = pd.DataFrame([eps, segmen_loss_l, domain_loss_l, train_accuracy_l, val_accuracy, val_accuracy_target, val_disc_accu])

    training_list.to_csv('Training_loop.csv')

    torch.save(network, 'LastDANNModel.pt')

    return best_model_f1, target_f1, best_overall, best_network, training_list


    #     if epoch == 0:
    #         # Evaluate network on validation dataset
    #         f1_val, loss_val = evaluate(network, source_val_loader, seg_loss, accu_function, Love, binary_love)
    #         val_accuracy.append(f1_val)
        
    #         f1_val_target, loss_val_target = evaluate(network, target_val_loader, seg_loss, accu_function, Love, binary_love)
    #         val_accuracy_target.append(f1_val_target)
        
    #     for k in range(int(batch_iterations)):
            
    #         if source_bigger_target:
                
    #             if (k == int(batch_iterations)-1): # For the last batch since source and target dataset might not match in dimensions.
    #                 temp_S_DS = torch.utils.data.Subset(source_train_dataset, [i for i in np.arange(int(k*len(target_train_dataset)), len(source_train_dataset), 1)])
    #             else:
    #                 temp_S_DS = torch.utils.data.Subset(source_train_dataset, [i for i in np.arange(int(k*len(target_train_dataset)), int((k+1)*len(target_train_dataset)), 1)])
        
    #             # Create train data loaders
    #             S_loader = torch.utils.data.DataLoader(dataset=temp_S_DS, batch_size=batch_size//2, shuffle=True)
    #             T_loader = torch.utils.data.DataLoader(dataset=target_train_dataset, batch_size=batch_size//2, shuffle=True)

    #         else:

    #             if (k == int(batch_iterations)-1): # For the last batch since source and target dataset might not match in dimensions.
    #                 temp_T_DS = torch.utils.data.Subset(target_train_dataset, [i for i in np.arange(int(k*len(source_train_dataset)), len(target_train_dataset), 1)])
    #             else:
    #                 temp_T_DS = torch.utils.data.Subset(target_train_dataset, [i for i in np.arange(int(k*len(source_train_dataset)), int((k+1)*len(source_train_dataset)), 1)])
        
    #             # Create train data loaders
    #             S_loader = torch.utils.data.DataLoader(dataset=source_train_dataset, batch_size=batch_size//2, shuffle=True)
    #             T_loader = torch.utils.data.DataLoader(dataset=temp_T_DS, batch_size=batch_size//2, shuffle=True)

    #         batches = zip(S_loader, T_loader)
    #         n_batches = min(len(S_loader), len(T_loader))
    
    #         network.train()
    #         # discriminator.train()
    
    #         total_domain_loss = []
    #         oa = []
    #         total_segmentation_accuracy = []
    #         segment_loss = []
    
    #         iterable_batches = enumerate(batches)

    #         revgrad = np.max([0, (epoch - e_0)/(epochs - e_0)])
    #         # 2. / (1. + np.exp(-10 * epoch/epochs)) - 1 GANIN
    
    #         for j in range(n_batches):
    
    #             i, (source, target) = next(iterable_batches)

    #             if Love:
    #                 source_input = LOVE_resample_fly(source['image']).to(device)
    #                 target_input = LOVE_resample_fly(target['image']).to(device)  
    #                 source_GT = LOVE_resample_fly(source['mask']).type(torch.long).squeeze().to(device)
    #                 target_GT = LOVE_resample_fly(target['mask']).type(torch.long).squeeze().to(device)
                    
    #                 if binary_love:
    #                     source_GT = (source_GT == 6).long()
    #                     target_GT = (target_GT == 6).long()
    #             else:
    #                 source_input = source[0].to(device)
    #                 target_input = target[0].to(device)
    #                 source_GT = source[1].type(torch.long).squeeze().to(device)
    #                 target_GT = target[1].type(torch.long).squeeze().to(device)

    #             # Concatenate the input images from both domains
    #             input = torch.cat([source_input, target_input])
    
    #             # Get segmentation and domain ground truth labels
    #             seg_GT = source_GT
    #             domain_labels = torch.cat([torch.zeros(source_input.shape[0]),
    #                                         torch.ones(target_input.shape[0])]).to(device)
                
    #             # Get predictions
    #             features = network.FE(input)
    #             dw = network.FE.DownSteps(input)
                
    #             seg_preds = network.C(features, dw)
    #             dom_preds = network.D(features, revgrad)

    #             # Deal with incorrect dimensions
    #             if seg_preds[:source_input.shape[0]].max(1)[1].size() != seg_GT.size():
    #                 seg_GT = seg_GT[None, :, :]
    
    #             # Calculate the loss function
    #             segmentation_loss = seg_loss(seg_preds[:source_input.shape[0]], seg_GT)
    #             discriminator_loss = domain_loss(dom_preds.squeeze(), domain_labels)

    #             seg_imp = 1
                
    #             # Total loss
    #             loss = seg_imp*segmentation_loss + (2-seg_imp)*discriminator_loss
    
    #             # set the gradients of the model to 0 and perform the backward propagation
    #             optim.zero_grad()
    #             loss.backward()
    #             optim.step()
    
    #             f1 = accu_function.to(device)

    #             accu = f1(seg_preds[:source_input.shape[0]].max(1)[1], seg_GT)
                
    #             total_domain_loss.append(discriminator_loss.item())
    #             segment_loss.append(segmentation_loss.item())
    #             total_segmentation_accuracy.append(accu.item())
                
    #             oa.append(np.sum((dom_preds.squeeze().detach().cpu().numpy() > 0) == domain_labels.detach().cpu().numpy())/(domain_labels.detach().cpu().numpy().size))

    #             step = 2
    
    #             # Add loss and accuracy to total
    #             if ((j/(n_batches//step) == j//(n_batches//step))) & (j != 0):
                    
    #                 # total_segmentation_accuracy 
    #                 train_accuracy_l.append(np.nanmean(total_segmentation_accuracy))
    #                                         # /(n_batches//step))
        
    #                 # total_domain_loss 
    #                 domain_loss_l.append(np.nanmean(total_domain_loss))
    #                 # /(n_batches//step))
                
    #                 # segment_loss list
    #                 segmen_loss_l.append(np.nanmean(segment_loss))
    #                                      # /(n_batches//step))

    #                 train_disc_accuracy_l.append(np.nanmean(oa))
    #                 # (n_batches//step))

    #                 # Epochs list
    #                 eps.append(epoch + k/(batch_iterations) + (j/n_batches)/batch_iterations)

    #                 #Reset loss and accuracies
    #                 total_domain_loss = []
    #                 oa = []
    #                 total_segmentation_accuracy = []
    #                 segment_loss = []
    
    #     if (epoch//4 == epoch/4):
    #         #After 4 epochs, reduce the learning rate by a factor 
    #         optim.param_groups[0]['lr'] *= 0.75
        
    #     # Evaluate network on validation dataset
    #     f1_val, loss_val = evaluate(network, source_val_loader, seg_loss, accu_function, Love, binary_love, revgrad)
    #     val_accuracy.append(f1_val)
    
    #     f1_val_target, loss_val_target = evaluate(network, target_val_loader, seg_loss, accu_function, Love, binary_love, revgrad)
    #     val_accuracy_target.append(f1_val_target)

    #     oa_val = evaluate_disc(network, [source_val_loader, target_val_loader], device, Love)
        

    #     # Selection of best model so far using validation dataset.
    #     # Relative importance of segmentation over discrimination (0 to 5)
    #     rel_imp_seg = 4.75

    #     overall = ((5-rel_imp_seg)*(1-oa_val) + rel_imp_seg*(f1_val))/5

    #     print(oa_val, f1_val, overall)
        
    #     if epoch == 0:
    #         best_model_f1 = f1_val
    #         best_oa = oa_val
    #         best_overall = overall
    #         target_f1 = f1_val_target
    #         torch.save(network, 'BestDANNModel.pt')
    #         best_network = network
    #     else:
    #         if best_overall < overall:
    #             best_model_f1 = f1_val
    #             best_oa = oa_val
    #             best_overall = overall
    #             print(best_overall)
    #             target_f1 = f1_val_target
    #             torch.save(network, 'BestDANNModel.pt')
    #             best_network = network

    #     fig = plt.figure(figsize = (7,5))
        
    #     plt.plot(eps, segmen_loss_l, '-k', label = 'Segmentation loss')
    #     plt.plot(eps, domain_loss_l, '-r', label = 'Domain loss')
    #     plt.plot(eps, train_accuracy_l, '--g', label = 'Train segmentation accuracy')
    #     plt.plot(eps, train_disc_accuracy_l, '--y', label = 'Train discriminator accuracy')

    #     plt.ylim((0,1.5))

    #     fig.savefig('DANN_Training.png', dpi = 100)
    #     plt.close()
        
    # fig = plt.figure(figsize = (7,5))
        
    # plt.plot(eps, segmen_loss_l, '--k', label = 'Segmentation loss')
    # plt.plot(eps, domain_loss_l, '--r', label = 'Domain loss')
    # plt.plot(eps, train_accuracy_l, '--g', label = 'Train segmentation accuracy')
    # plt.plot(eps, train_disc_accuracy_l, '--y', label = 'Train discriminator accuracy')
    
    # plt.plot(np.arange(0, epochs + 1, 1), val_accuracy, 'darkblue', label = ' Validation Segmentation accuracy on source domain')
    # plt.plot(np.arange(0, epochs + 1, 1), val_accuracy_target, 'darkred', label = ' Validation Segmentation accuracy on target domain')

    # # plt.ylim((0,1))
    # plt.title('LR_Seg:' + str(learning_rate_seg) + ' LR_Disc:' + str(learning_rate_disc))
    
    # plt.legend()
    
    # plt.xlabel('Epochs')

    # fig.savefig('DANN_Training_'+ str(learning_rate_seg) + '_' + str(learning_rate_disc) + '.png', dpi = 150)

    # return best_model_f1, target_f1, best_overall, best_network
    