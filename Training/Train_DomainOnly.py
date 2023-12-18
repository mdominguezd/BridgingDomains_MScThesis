# import sys
# append a new directory to sys.path
# sys.path.append('../')
from collections import deque
import torch
from torchmetrics.classification import BinaryF1Score
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

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
    device = next(iter(net.parameters())).device # Get training device ("cuda" or "cpu")

    f1_scores = []
    losses = []

    with torch.no_grad():
        # Iterate over validate loader to get mean accuracy and mean loss
        for i, Data in enumerate(validate_loader):
            
            # The inputs and GT are obtained differently depending of the Dataset (LoveDA or our own DS)
            if Love:
                inputs = Data['image']
                GTs = Data['mask']
                
                # Make it a binary problem only detecting one class
                GTs[GTs != 2] = 0
                GTs[GTs == 2] = 1
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


def get_training_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def training_loop(network, train_loader, val_loader, learning_rate, starter_channels, momentum, number_epochs, loss_function, bilinear = True, n_channels = 4, n_classes = 2, plot = True, accu_function = BinaryF1Score(), seed = 8, Love = False):
    """
        Function to train the Neural Network.

        Input:
            - train_loader: DataLoader with the training dataset.
            - val_loader: DataLoader with the validation dataset.
            - learning_rate: Initial learning rate for training the network.
            - starter_channels: Starting number of channels in th U-Net
            - momentum: Momentum used during training.
            - number_epochs: Number of training epochs.
            - loss_function: Function to calculate loss
            - bilinear: Boolean to decide the upscaling method (If True Bilinear if False Transpose convolution. Default: True)
            - n_channels: Number of initial channels (Defalut 4 [Planet])
            - n_classes: Number of classes that will be predicted (Default 2 [Binary segmentation])
            - plot: Boolean to decide if training loop should be plotted or not.
            - accu_function: Function to calculate accuracy (Default: BinaryF1Score)
            - seed: Seed that will be used for generation of random values.
            - Love: Boolean to decide between training with LoveDA dataset or our own dataset.

        Output:
            - best_model: f1-score of the best model trained. (Calculated on validation dataset) 
            - model_saved: The best model trained.
            - spearman: Spearman correlation calculated for training progress (High positive value will indicate positive learning)
    """
    
    device = get_training_device()

    np.random.seed(seed) 
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    network = network
    network.to(device)
    optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate, momentum = momentum)
    
    #Training metrics are computed as a running average of the last x samples
    loss_train = deque(maxlen=len(train_loader))
    accuracy_train = deque(maxlen=len(train_loader))

    val_eps = []
    val_f1s = []
    val_loss = []

    train_eps = []
    train_f1s = []
    train_loss = []
    
    for epoch in range(number_epochs):
    
        #Validation phase 1:
        metric_val = evaluate(network, val_loader, loss_function, accu_function, Love)

        val_eps.append(epoch)
        val_f1s.append(metric_val[0])
        val_loss.append(metric_val[1])
            
        #Training phase:
        network.train() #indicate to the network that we enter training mode
        
        for i, Data in enumerate(train_loader): # Iterate over the training dataset and do the backward propagation.
            if Love:
                inputs = Data['image']
                GTs = Data['mask']
                GTs[GTs != 2] = 0
                GTs[GTs == 2] = 1
            else:
                inputs = Data[0]
                GTs = Data[1]
                
            inputs = inputs.to(device)
            GTs = GTs.type(torch.long).squeeze().to(device)
            
            #Set the gradients of the model to 0.
            optimizer.zero_grad()
            pred = network(inputs)

            if (pred.max(1)[1].shape != GTs.shape):
                GTs = GTs[None, :, :]
            
            loss = loss_function(pred, GTs)
            
            f1 = accu_function.to(device)
            overall_accuracy = f1(pred.max(1)[1], GTs)
            #We accumulate the gradients...
            loss.backward()
            #...and we update the parameters according to the gradients.
            optimizer.step()
            loss_train.append(loss.item()/GTs.shape[0])
            accuracy_train.append(overall_accuracy.item())

            train_eps.append(epoch+i/len(train_loader))
            train_f1s.append(np.mean(accuracy_train))
            train_loss.append(np.mean(loss_train))

        #Validation phase 1:
        metric_val = evaluate(network, val_loader, loss_function, accu_function, Love)

        val_eps.append(epoch + 1)
        val_f1s.append(metric_val[0])
        val_loss.append(metric_val[1])
        
        if epoch == 0:
            best_model = metric_val[0]
            torch.save(network, 'BestModel.pt')
            model_saved = network
        else:
            if best_model < metric_val[0]:
                best_model = metric_val[0]
                torch.save(network, 'BestModel.pt')
                model_saved = network
        
        if (epoch//4 == epoch/4):
            #After 4 epochs, reduce the learning rate by a factor of 0.2
            optimizer.param_groups[0]['lr'] *= 0.5

    spearman = stats.spearmanr(train_eps, train_f1s)[0]
    
    if plot:
        fig, ax = plt.subplots(1,1, figsize = (7,5))

        ax.plot(train_eps, train_f1s, label = 'Training F1-Score', ls= '--', color = 'r')
        ax.plot(train_eps, train_loss, label = 'Training Loss', ls = '-', color = 'r')

        ax.plot(val_eps, val_f1s, label = 'Validation F1-Score', ls = '--', color = 'b')
        ax.plot(val_eps, val_loss, label = 'Validation Loss', ls = '-', color = 'b')
        
        ax.text(val_eps[np.argmax(val_f1s)], np.max(val_f1s), str(np.max(val_f1s)))

        ax.set_xlabel("Epoch")

        plt.legend()

        fig.savefig('TrainingLoop.png', dpi = 200)
        
    return best_model, model_saved, spearman