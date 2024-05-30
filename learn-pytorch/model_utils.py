import numpy as np
import torch
import torch.nn as nn

def make_train_step_fn(model, loss_func, optimizer):
    
    # Builds function that performs a step in the training loop
    def perform_train_step_fn(x, y):
        # sets the model in training mode
        model.train()
        
        # step-1: compute the forward pass - the model's prediction
        y_hat = model(x)
        
        # step-2: compute loss
        loss = loss_func(y_hat, y)
        
        # step-3: compute gradients
        loss.backward()
        
        # step-4: update model params
        optimizer.step()
        optimizer.zero_grad()
        
        # extract params (optional)
        # b = model.state_dict()['linear.bias'].item()
        # w = model.state_dict()['linear.weight'].item()
        
        # return the loss
        return loss.item()
    
    # return the function that will be called inside the training loop
    return perform_train_step_fn

def make_val_step_fn(model, loss_func):
    
    # Builds function that performs a step in the training loop
    def perform_val_step_fn(x, y):
        # sets the model in evaluation mode
        model.eval()
        
        # step-1: compute the forward pass - the model's prediction
        y_hat = model(x)
        
        # step-2: compute loss
        loss = loss_func(y_hat, y)
        
        # extract params (optional)
        # b = model.state_dict()['linear.bias'].item()
        # w = model.state_dict()['linear.weight'].item()
        
        # return the loss
        return loss.item()
    
    # return the function that will be called inside the training loop
    return perform_val_step_fn

def mini_batch(device, data_loader, step_fn):
    mini_batch_losses = []
    
    for x_batch, y_batch in data_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        mini_batch_loss = step_fn(x_batch, y_batch)
        mini_batch_losses.append(mini_batch_loss)
    
    mini_batch_losses.append(mini_batch_loss)
    loss = np.mean(mini_batch_losses)
    
    return loss