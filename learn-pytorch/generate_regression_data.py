import numpy as np
import pandas as pd
from torch.utils.data import Dataset


def generate_simple_regression_data(true_bias, true_weight, sample_size=1000, mode='train', set_seed=42):
    
    x = np.random.rand(sample_size, 1)
    epsilon = (0.1 * np.random.randn(sample_size, 1))
    y = true_bias + (true_weight * x) + epsilon

    # shuffle the idx
    idx = np.arange(sample_size)
    np.random.shuffle(idx)

    # Uses first 80 random indices for train
    train_idx = idx[:int(sample_size * 0.8)]

    # Uses the remaining indices for validation
    val_idx = idx[int(sample_size * 0.8):]

    # Generates train and validation sets
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    
    return x_train, y_train, x_val, y_val

class CustomDataset(Dataset):
    """A Class to create custom dataset objects"""
    def __init__(self, x_tensor, y_tensor):
        super(Dataset).__init__()
        self.x = x_tensor
        self.y = y_tensor
        
    def __getitem__(self, index):
        return (self.x[index], self.y[index])
    
    def __len__(self):
        return len(self.x)
    
if __name__ == "__main__":
    pass
    