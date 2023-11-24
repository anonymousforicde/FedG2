
import torch
import torch.optim as optim
import torch.nn as nn


def train_sgd(gpu, model, criterion, optimizer, train_set):  
    model.train()
    train_loss = 0.0    

    optimizer.zero_grad()
    for step, (data, label) in enumerate(train_set): 
        data   = data.cuda(gpu, non_blocking=True)
        label  = label.cuda(gpu, non_blocking=True)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)                                         
        loss.backward()     
        optimizer.step()
        train_loss += loss.item()
    
    return train_loss



def get_global_grad(global_model, val_set, lr):
    global_optim = optim.SGD(global_model.parameters(), lr=lr)
    criterion    = nn.CrossEntropyLoss().cuda()
    
    total_loss = 0.0
    global_optim.zero_grad()
    for step, (data, label) in enumerate(val_set): 
        output = global_model(data.cuda())
        loss   = criterion(output, label.cuda())
        total_loss += loss.item()
        loss.backward()
    return global_optim, total_loss

