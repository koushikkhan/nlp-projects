import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class GenerateRandomDataset():
    def __init__(self):
        pass
    
    @staticmethod
    def generate_slr_dataset_v1(true_bias, true_weight, sample_size=1000, train_perc=0.8, set_seed=42):
        # this method partitions the numpy arrays
        
        x = np.random.rand(sample_size, 1)
        epsilon = (0.1 * np.random.randn(sample_size, 1))
        y = true_bias + (true_weight * x) + epsilon

        # shuffle the idx
        idx = np.arange(sample_size)
        np.random.shuffle(idx)

        # Uses first `train_perc` percentage random indices for train
        train_idx = idx[:int(sample_size * train_perc)]

        # Uses the remaining indices for validation
        val_idx = idx[int(sample_size * train_perc):]

        # Generates train and validation sets
        x_train, y_train = x[train_idx], y[train_idx]
        x_val, y_val = x[val_idx], y[val_idx]

        return x_train, y_train, x_val, y_val
    
    @staticmethod
    def generate_slr_dataset_v2(true_bias, true_weight, sample_size=1000, set_seed=13):
        # this method only returns the numpy arrays
        # partioning will be done by pytorch's random_split() method
        
        x = np.random.rand(sample_size, 1)
        epsilon = (0.1 * np.random.randn(sample_size, 1))
        y = true_bias + (true_weight * x) + epsilon

        return x, y
    
    @staticmethod
    def generate_classification_dataset():
        pass
    

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
    