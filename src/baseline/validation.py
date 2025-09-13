import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt


class DataValidator():

    def __init__(self, n_splits:int = 3, max_val_size:int = None, val_size:int = None, gap:int = 0):
        self.n_splits = n_splits
        self.max_val_size = max_val_size
        self.val_size = val_size
        self.gap = gap

    def get_n_splits(self, X = None, y = None, groups = None):
        return self.n_splits
    

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        val_size = self.val_size or (n_samples // (self.n_splits + 1))
        train_size = n_samples - val_size

        for i in range(self.n_splits):
            val_end = train_size + (i * val_size)
            val_start = val_end + self.gap
            val_end = val_start + val_size

            train_indices = np.arange(val_end)
            val_indices = np.arange(val_start, val_end)

            yield train_indices, val_indices