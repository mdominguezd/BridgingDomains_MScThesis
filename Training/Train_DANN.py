target_loader = TNZ_train_loader
source_loader = CIV_train_loader

#This should go to Dataset module

CIV_train_DS = Img_Dataset('IvoryCoast', transform, norm = normalization, VI  =Use_VI)
CIV_val_DS = Img_Dataset('IvoryCoast', split = 'Validation', norm = normalization, VI  =Use_VI)
CIV_test_DS = Img_Dataset('IvoryCoast', split = 'Test', norm = normalization, VI  =Use_VI)

TNZ_train_DS = Img_Dataset('Tanzania', transform, norm = normalization, VI  =Use_VI)
TNZ_val_DS = Img_Dataset('Tanzania', split = 'Validation', norm = normalization, VI  =Use_VI)
TNZ_test_DS = Img_Dataset('Tanzania', split = 'Test', norm = normalization, VI  =Use_VI)

CIV_n_batches = np.ceil(len(CIV_train_DS)/(batch_size/2))
TNZ_n_batches = np.ceil(len(TNZ_train_DS)/(batch_size/2))

n_batches = min(CIV_n_batches, TNZ_n_batches)

batch_iterations = np.ceil(max(CIV_n_batches, TNZ_n_batches) / n_batches)

def initialize_unet(n_channels, n_classes, bilinear, starter, up_layer):
    """
        Function to initialize U-Net and the discriminator that will be trained using UNet-DANN
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    network = UNet(n_channels=n_channels, n_classes=n_classes, bilinear = bilinear, starter = starter, up_layer = up_layer).to(device)

    in_feat = 16**2 * st*(2**3) # Number of features that go in the fully connected layers depends on the number of starting channels
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    discriminator = disc(in_feat, bilinear, starter, up_layer).to(device)

    return network, discriminator

def evaluate(net, validate_loader):
    #You need to set the network to eval mode when using batch normalization (to
    #be consistent across evaluation samples, we use mean and stddev computed
    #during training when doing inference, as opposed to ones computed on the
    #batch) or dropout (you want to use all the parameters during inference).
    net.eval()  # Set the model to evaluation mode
    device = next(iter(net.parameters())).device
    
    f1_scores = []
    
    for i, (inputs, GTs) in enumerate(validate_loader):
        inputs = inputs.to(device)
        GTs = GTs.type(torch.long).squeeze().to(device)
        pred = net(inputs)
        
        f1 = BinaryF1Score().to(device)
        f1_score = f1(pred.max(1)[1], GTs)
        f1_scores.append(f1_score.to('cpu').numpy())
    
    return np.mean(f1_scores)

def DANN_training_loop(network, disc, source_train_loader, target_train_loader, source_val_loader, target_val_loader, epochs):
    """
        Function to create the training loop for UNet-DANN
    """

    segmen_loss_l = []
    train_accuracy_l = []
    domain_loss_l = []
    val_f1s = []
    val_f1s_target = []
    
    for epoch in range(epochs):

        T_DS = TNZ_train_DS
    
        for k in range(int(batch_iterations)):
    
            if (k == int(batch_iterations)-1): # For the last batch since
                temp_S_DS = torch.utils.data.Subset(CIV_train_DS, [i for i in np.arange(int(k*len(TNZ_train_DS)), len(CIV_train_DS), 1)])
            else:
                temp_S_DS = torch.utils.data.Subset(CIV_train_DS, [i for i in np.arange(int(k*len(TNZ_train_DS)), int((k+1)*len(TNZ_train_DS)), 1)])
    
            # Create train data loaders
            S_loader = torch.utils.data.DataLoader(dataset=temp_S_DS, batch_size=batch_size//2, shuffle=True)
            T_loader = torch.utils.data.DataLoader(dataset=T_DS, batch_size=batch_size//2, shuffle=True)
    
            batches = zip(S_loader, T_loader)
    
            n_batches = min(len(S_loader), len(T_loader))
    
            network.train()
            discriminator.train()
    
            total_domain_loss = total_segmentation_accuracy = segment_loss = 0
    
            iterable_batches = enumerate(batches)
    
            for k in range(n_batches):
    
                i, (source, target) = next(iterable_batches)
    
                source_input = source[0].to(device)
                target_input = target[0].to(device)
    
                input = torch.cat([source_input, target_input])
    
                source_GT = source[1].type(torch.long).squeeze().to(device)
                target_GT = target[1].type(torch.long).squeeze().to(device)
    
                # Calculate segmentation and domain groung truth labels
                seg_GT = source_GT
                domain_labels = torch.cat([torch.zeros(source_input.shape[0]),
                                            torch.ones(target_input.shape[0])]).to(device)
    
                # Get predictions
                features = network.FE(input)
                dw = network.FE.DownSteps(input)
                seg_preds = network.C(features, dw)
                dom_preds = discriminator(features)
    
                # Calculate the loss function
                segmentation_loss = seg_loss(seg_preds[:source_input.shape[0]], seg_GT)
                discriminator_loss = domain_loss(dom_preds.squeeze(), domain_labels)
    
                loss = discriminator_loss + segmentation_loss
    
                # Perform the backward propagation
                optim.zero_grad()
                loss.backward()
                optim.step()
    
                f1 = BinaryF1Score().to(device)
    
                # Add loss and accuracy to total
                total_domain_loss += discriminator_loss.item()
                segment_loss += segmentation_loss.item()
                total_segmentation_accuracy += f1(seg_preds[:source_input.shape[0]].max(1)[1], seg_GT).item()

        # Evaluate network on validation dataset
        f1_val = evaluate(network, source_val_loader)
        val_f1s.append(f1_val)
    
        f1_val_target = evaluate(network, target_val_loader)
        val_f1s_target.append(f1_val_target)
    
        if epoch == 0:
            best_model_f1 = f1_val
            torch.save(network, 'BestDANNModel.pt')
        else:
            if best_model_f1 < f1_val:
                best_model_f1 = f1_val
                torch.save(network, 'BestDANNModel.pt')
    
        total_domain_loss /= n_batches
        domain_loss_l.append(total_domain_loss)
    
        segment_loss /= n_batches
        segmen_loss_l.append(segment_loss)
    
        total_segmentation_accuracy /= n_batches
        train_accuracy_l.append(total_segmentation_accuracy)

    