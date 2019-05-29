# -*- coding: utf-8 -*-
"""
Created on Mon May 27 23:12:18 2019

@author: Guilherme
"""
import itertools
import numpy as np
import copy
from Processor import Processor

class CrossProcessor:
    
    def __init__(self, training, testing, target_var, random_state, verbose):
        
        self.training = training
        self.testing = testing
        self.target_var = target_var
        self.random_state = random_state
        self.verbose = verbose
        self.stages = []
        self.ite_c = 0
        
    def addStage(self, stage):
        self.stages.append(stage)
    
    def addStages(self, stages):
        self.stages = stages
        
    def mixer(self):
        combinations = []
        for stage in self.stages:
            combinations.append(np.arange(len(stage.functions)))
            
        self.combinations = list(itertools.product(*combinations))
    
    def process(self):
        processor = Processor(self.training, self.testing, self.target_var, self.random_state)
        processor.addStages(copy.deepcopy(self.stages), self.combinations[self.ite_c])
        processor.exec_()
        self.ite_c += 1
        return processor
    #Missing Values Treatment
    #   Drop missing values/ Impute missing values
    
    #Outlier Detection
    #   Zscore/Mahalanobis
    
    #Outlier Treatment
    #   Smoothing/Removal
    
    #Feature Selection
    #   RFE/PCA