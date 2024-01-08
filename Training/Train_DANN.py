import torch
import matplotlib.pyplot as plt
from torchmetrics.classification import BinaryF1Score
from tqdm import tqdm

from Dataset.ReadyToTrain_DS import *
from Models.U_Net import *

plt.style.use('ggplot')

def get_training_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def initialize_Unet_DANN(n_channels, n_classes, bilinear = True, starter = 16, up_layer = 4, attention = True, Love = False):
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
    
    seed = 123
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    network = UNet(n_channels=n_channels, n_classes=n_classes, bilinear = bilinear, starter = starter, up_layer = up_layer, attention = attention).to(device)

    # Calculate the number  of features that go in the fully connected layers of the discriminator
    if Love:
        img_size = 1024
    else:
        img_size = 256
        
    in_feat = (img_size//(2**4))**2 * starter*(2**3) 
        
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    discriminator = disc(in_feat, bilinear, starter, up_layer).to(device)

    return network, discriminator

def evaluate(net, validate_loader, loss_function, accu_function = BinaryF1Score(), Love = False):
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
                inputs = Data['image']
                GTs = Data['mask']
            else:
                inputs = Data[0]
                GTs = Data[1]
        
            inputs = inputs.to(device)
            GTs = GTs.type(torch.long).squeeze().to(device)
            pred = net(inputs)
        
            f1 = accu_function.to(device)
        
            if (pred.max(1)[1].shape != GTs.shape):
                GTs = GTs[None, :, :]

            loss = loss_function(pred, GTs)/GTs.shape[0]
        
            f1_score = f1(pred.max(1)[1], GTs)
            
            f1_scores.append(f1_score.to('cpu').numpy())
            losses.append(loss.to('cpu').numpy())

        metric = [np.mean(f1_scores), np.mean(losses)]   
        
    return metric

def DANN_training_loop(source_domain, target_domain, DS_args, network_args, learning_rate, momentum, epochs, Love = False, seg_loss = torch.nn.CrossEntropyLoss(), domain_loss = torch.nn.BCEWithLogitsLoss(), accu_function = BinaryF1Score()):
    """
        Function to carry out the training loop for UNet-DANN.

        Inputs:
            - network: U-Net architecture to be trained.
            - disc: Discriminator that will be trained. 
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
    source_val_loader = torch.utils.data.DataLoader(dataset=source_DSs[1], batch_size=batch_size, shuffle=True)
    target_val_loader = torch.utils.data.DataLoader(dataset=target_DSs[1], batch_size=batch_size, shuffle=True)
    
    # Initialize the networks to be trained and the optimizer
    network, discriminator = initialize_Unet_DANN(n_channels, *network_args)
    optim = torch.optim.SGD(list(discriminator.parameters()) + list(network.parameters()), lr=learning_rate, momentum = momentum)
    
    # Create empty lists where segementation accuracy in source dataset and segmentation and domain loss will be stored.
    val_accuracy = []
    val_accuracy_target = []
    segmen_loss_l = []
    train_accuracy_l = []
    domain_loss_l = []

    eps = []

    for epoch in tqdm(range(epochs), desc = 'Training UNet-DANN model'):

        if epoch == 0:
            # Evaluate network on validation dataset
            f1_val, loss_val = evaluate(network, source_val_loader, seg_loss, accu_function, Love)
            val_accuracy.append(f1_val)
        
            f1_val_target, loss_val_target = evaluate(network, target_val_loader, seg_loss, accu_function, Love)
            val_accuracy_target.append(f1_val_target)
        
        for k in range(int(batch_iterations)):

            if source_bigger_target:
                
                if (k == int(batch_iterations)-1): # For the last batch since source and target dataset might not match in dimensions.
                    temp_S_DS = torch.utils.data.Subset(source_train_dataset, [i for i in np.arange(int(k*len(target_train_dataset)), len(source_train_dataset), 1)])
                else:
                    temp_S_DS = torch.utils.data.Subset(source_train_dataset, [i for i in np.arange(int(k*len(target_train_dataset)), int((k+1)*len(target_train_dataset)), 1)])
        
                # Create train data loaders
                S_loader = torch.utils.data.DataLoader(dataset=temp_S_DS, batch_size=batch_size//2, shuffle=True)
                T_loader = torch.utils.data.DataLoader(dataset=target_train_dataset, batch_size=batch_size//2, shuffle=True)

            else:

                if (k == int(batch_iterations)-1): # For the last batch since source and target dataset might not match in dimensions.
                    temp_T_DS = torch.utils.data.Subset(target_train_dataset, [i for i in np.arange(int(k*len(source_train_dataset)), len(target_train_dataset), 1)])
                else:
                    temp_T_DS = torch.utils.data.Subset(target_train_dataset, [i for i in np.arange(int(k*len(source_train_dataset)), int((k+1)*len(source_train_dataset)), 1)])
        
                # Create train data loaders
                S_loader = torch.utils.data.DataLoader(dataset=source_train_dataset, batch_size=batch_size//2, shuffle=True)
                T_loader = torch.utils.data.DataLoader(dataset=temp_T_DS, batch_size=batch_size//2, shuffle=True)

            batches = zip(S_loader, T_loader)
            n_batches = min(len(S_loader), len(T_loader))
    
            network.train()
            discriminator.train()
    
            total_domain_loss = total_segmentation_accuracy = segment_loss = 0
    
            iterable_batches = enumerate(batches)
    
            for j in range(n_batches):
    
                i, (source, target) = next(iterable_batches)

                if Love:
                    source_input = source['image'].to(device)
                    target_input = target['image'].to(device)  
                    source_GT = source['mask'].type(torch.long).squeeze().to(device)
                    target_GT = target['mask'].type(torch.long).squeeze().to(device)
                else:
                    source_input = source[0].to(device)
                    target_input = target[0].to(device)
                    source_GT = source[1].type(torch.long).squeeze().to(device)
                    target_GT = target[1].type(torch.long).squeeze().to(device)
    
                input = torch.cat([source_input, target_input])
    
                # Calculate segmentation and domain ground truth labels
                seg_GT = source_GT
                domain_labels = torch.cat([torch.zeros(source_input.shape[0]),
                                            torch.ones(target_input.shape[0])]).to(device)
    
                # Get predictions
                features = network.FE(input)
                dw = network.FE.DownSteps(input)
                seg_preds = network.C(features, dw)
                dom_preds = discriminator(features)

                if seg_preds[:source_input.shape[0]].max(1)[1].size() != seg_GT.size():
                    seg_GT = seg_GT[None, :, :]
    
                # Calculate the loss function
                segmentation_loss = seg_loss(seg_preds[:source_input.shape[0]], seg_GT)
                discriminator_loss = domain_loss(dom_preds.squeeze(), domain_labels)

                # Total loss
                loss = 1*discriminator_loss + 1*segmentation_loss
    
                # Perform the backward propagation
                optim.zero_grad()
                loss.backward()
                optim.step()
    
                f1 = accu_function.to(device)

                total_domain_loss += discriminator_loss.item()
                segment_loss += segmentation_loss.item()
                total_segmentation_accuracy += f1(seg_preds[:source_input.shape[0]].max(1)[1], seg_GT).item()

                step = 4
    
                # Add loss and accuracy to total
                if (j/(n_batches//step) == j//(n_batches//step)):

                    if j == 0:
                        step = n_batches
                    
                    # total_segmentation_accuracy 
                    train_accuracy_l.append(total_segmentation_accuracy/(n_batches//step))
        
                    # total_domain_loss 
                    domain_loss_l.append(total_domain_loss/(n_batches//step))
                
                    # segment_loss 
                    segmen_loss_l.append(segment_loss/(n_batches//step))

                    # Epochs list
                    eps.append(epoch + k/(batch_iterations) + (j/n_batches)/batch_iterations)

                    #Reset loss and accuracies
                    total_domain_loss = total_segmentation_accuracy = segment_loss = 0

        # Evaluate network on validation dataset
        f1_val, loss_val = evaluate(network, source_val_loader, seg_loss, accu_function, Love)
        val_accuracy.append(f1_val)
    
        f1_val_target, loss_val_target = evaluate(network, target_val_loader, seg_loss, accu_function, Love)
        val_accuracy_target.append(f1_val_target)
    
        if epoch == 0:
            best_model_f1 = f1_val
            torch.save(network, 'BestDANNModel.pt')
            best_network = network
        else:
            if best_model_f1 < f1_val:
                best_model_f1 = f1_val
                torch.save(network, 'BestDANNModel.pt')
                best_network = network

        fig = plt.figure(figsize = (7,5))
        
        plt.plot(eps, segmen_loss_l, '--k', label = 'Segmentation loss')
        plt.plot(eps, domain_loss_l, '--r', label = 'Domain loss')
        plt.plot(eps, train_accuracy_l, '--g', label = 'Train segmentation accuracy')

        # plt.ylim((0,1))
    
        plt.legend()
        
        plt.xlabel('Epochs')

        plt.title('LR: '+ str(learning_rate))
        
        fig.savefig('DANN_Training.png', dpi = 200)
    
    plt.plot(np.arange(0, epochs + 1, 1), val_accuracy, 'darkblue', label = ' Validation Segmentation accuracy on source domain')
    plt.plot(np.arange(0, epochs + 1, 1), val_accuracy_target, 'darkred', label = ' Validation Segmentation accuracy on target domain')

    plt.ylim((0,1))
    
    plt.legend()
    
    plt.xlabel('Epochs')

    plt.title('LR: '+ str(learning_rate))

    fig.savefig('DANN_Training.png', dpi = 200)

    return best_model_f1, best_network


    