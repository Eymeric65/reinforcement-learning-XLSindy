"""
This set of class are made to setup a place holder for the observation and action spaces
"""

import numpy as np
import warnings

# class Spaces:
#     def __init__(self):
        
#         self.size = None


#     def boop(self):

#         print("Boop")

class BoxArray:
    def __init__(self, shape,low = None, high= None):
        self.low = low
        self.high = high
        self.shape = shape

    def sample(self):
        if self.low or self.high is None:
            warnings.warn("Sampling between -1 and 1 as low or high is None", UserWarning)
            return np.random.uniform(-1, 1, self.shape)
        
        return np.random.uniform(self.low, self.high, self.shape)
    
    def contains(self, x):
        if self.low or self.high is None:
            warnings.warn("Contains between -1 and 1 as low or high is None", UserWarning)
            return np.all(x >= -1) and np.all(x <= 1)
        return np.all(x >= self.low) and np.all(x <= self.high)
    


