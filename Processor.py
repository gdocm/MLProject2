# -*- coding: utf-8 -*-
"""
Created on Mon May 27 23:33:36 2019

@author: Guilherme
"""
import numpy as np
class Processor:
    
    def __init__(self, training, testing, target_var, random_state):
        self.stages = []
        self.training = training
        self.testing = testing
        self.target_var = target_var
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        self.numerical_vars = self.training.select_dtypes(include = numerics).columns
        self.categorical_vars = list(set(self.training.columns) - set(self.numerical_vars))
    
    def addStages(self,stages,comb = (0,0,0)):
        for i in range(len(stages)):
            stages[i].processor = self
            stages[i].functions =   [stages[i].functions[comb[i]]]
            temp = [stages[i].processor]
            for p in stages[i].params[comb[i]]:
                temp.append(p)
            stages[i].params = [temp]
            print(stages[i].params)
        self.comb = comb
        self.stages = stages

    def exec_(self):
        for i in range(len(self.stages)):
            stage = self.stages[i]
            if len(stage.functions) > 0:
                self = stage.functions[0](*stage.params[0])