import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np 

# function to train model
def train(model, loader, loss_func, optimizer, device):
    model.train()

    running_loss, running_accuracy = 0.0, 0.0
    num_samples = 0

    for i, (img, lbl) in enumerate(loader):
        img, lbl = img.to(device), lbl.to(device)
        # compute the forward pass through
        out = model(img)
        loss = loss_func(out, lbl)
        
        running_loss += loss.item() * img.shape[0]
        num_samples += img.shape[0]
        
        pred = out.argmax(dim=1)
        running_accuracy += (pred==lbl).sum().item()

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return running_loss/num_samples, running_accuracy/num_samples

# function to test the model
def test(model, loader, loss_func, device):
    # Turn off gradients for validation, saves memory and computation
    with torch.no_grad(): 
        label = []
        predicted = []
        model.eval() # switch to evaluation mode
        running_loss, running_accuracy = 0.0, 0.0
        num_samples = 0
        for i, (img, lbl) in enumerate(loader):
            # copy data to GPU if any
            img, lbl = img.to(device), lbl.to(device)
            # compute forward pass
            out = model(img)
            num_samples += img.shape[0]
            
            running_loss += img.shape[0] * loss_func(out, lbl).item() # loss function averaging over its samples

            pred = out.argmax(dim=1)
            running_accuracy += (pred == lbl).sum().item()
            
            label.extend(lbl.tolist())
            predicted.extend(pred.tolist())

        return running_loss/num_samples, running_accuracy/num_samples, label, predicted


# train model on a certain number of epoch and upon a defined learning rate. Return train and validation loss and accuracy history
def model_run(model, train_loader, val_loader, epoch, lr, DEVICE, weight_decay=None, l2reg=False):
    loss_func = nn.CrossEntropyLoss()
    if l2reg:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    train_loss_h = []
    train_acc_h = []
    val_loss_h = []
    val_acc_h = []
    
    test_label = []
    test_predicted = []

    # DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")

    for it in range(epoch):
        train_loss, train_acc = train(model, train_loader, loss_func, optimizer, DEVICE)
        val_loss, val_acc, lbl, pred = test(model, val_loader, loss_func, DEVICE)

        train_loss_h.append(train_loss)
        train_acc_h.append(train_acc)
        val_loss_h.append(val_loss)
        val_acc_h.append(val_acc)

        #test_label.append(lbl)
        #test_predicted.append(pred)

        print('Epoch {}'.format(it))
        print('Training dataset: loss = {:.4f} || accuracy = {:.4f}'.format(train_loss, train_acc))
        print('Validation dataset: loss = {:.4f} || accuracy = {:.4f}'.format(val_loss, val_acc))
    
    test_label = np.copy(lbl)
    test_predicted = np.copy(pred)
    

    return train_loss_h, train_acc_h, val_loss_h, val_acc_h, test_label, test_predicted


