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
        b = model.state_dict()['linear.bias'].item()
        w = model.state_dict()['linear.weight'].item()
        
        # return the loss
        return loss.item(), b, w
    
    # return the function that will be called inside the training loop
    return perform_train_step_fn