import sys
import pandas as pd
from datetime import datetime, date
from sklearn.preprocessing import OneHotEncoder
import numpy as np
class Dataset:
    """ Loads and prepares the data

        The objective of this class is load the dataset and execute basic data
        preparation before effectively moving into the cross validation workflow.

    """

    def __init__(self, training,testing,target_variable):
        self.training = training
        self.testing = testing
        self.target_variable = target_variable
        self._drop_duplicates()

    def _drop_duplicates(self):
        print(self.training.shape)
        self.training.drop_duplicates(inplace = True)
        self.training.drop_duplicates(subset = self.training.drop(self.target_variable, axis = 1).columns , keep = False, inplace = True)
        print(self.training.shape)
