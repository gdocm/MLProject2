# -*- coding: utf-8 -*-
"""
Created on Mon May 27 23:17:17 2019

@author: Guilherme
"""
from feature_engineering import FeatureEngineer
from gplearn_MLAA.genetic import SymbolicRegressor
import numpy as np

class Stage:
    
    def __init__(self,name):
        
        self.name = name
        self.functions = []
        self.params = []
        self.processor = None
        
    def addMethod(self, function, params):
        '''Add Method options for stage'''
        if self.processor == None:
            temp = []
        else:
            temp = [self.processor]
        for param in params:
            temp.append(param)
        params = temp
        self.functions.append(function)
        self.params.append(params)
