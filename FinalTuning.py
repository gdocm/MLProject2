# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 12:39:34 2019

@author: Guilherme
"""

from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor, AdaBoostRegressor
from gplearn_MLAA.genetic import SymbolicRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error,mean_absolute_error
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
import itertools
import copy
import pickle

### MODELS
class model_runner():
        
    def __init__(self,training,labels,testing,seed,metric = 'neg_mean_absolute_error',cv = 5):
        
        self.training = training.drop(labels, axis = 1)
        self.labels = training[labels]
        self.testing = testing.drop(labels, axis = 1)
        self.labels_t = testing[labels]
        self.seed = seed
        self.metric = metric
        self.cv = cv
        self.best_params = []



            
        #Hyper Parameter Optimization
        hue_initialization_params = [True]*2
        selection = ['semantic_tournament']*2
        p_selective_crossover  =[0.1]*2
        p_grasm_mutation = [0.9]*2
        rs = [self.seed]*2
        dynamic_depth = [False,True]
        param_grid_gp = {
               'hue_initialization_params':hue_initialization_params,
               'selection':selection,
               'p_selective_crossover':p_selective_crossover,
               'p_grasm_mutation':p_grasm_mutation,
               'dynamic_depth':dynamic_depth,
               'random_state':rs}
        
        self.gridSearchGp(param_grid_gp)
    
    def gridSearchGp(self,param_grid):
        parameters = list(param_grid.values())
        comb = []
        for i in range(len(parameters[0])):
            t =  {}
            for j in param_grid.keys():
                t[j] = param_grid[j][i]
            comb.append(t)
        gp_results = {}
        for c in range(len(comb)):
            gp_results[c] = []
        for c in range(len(comb)):

            est_gp = SymbolicRegressor(**comb[c])
            est_gp.fit(self.training, self.labels)
            preds = est_gp.predict(self.testing)
            gp_results[c] = (est_gp.recorder, mean_absolute_error(self.labels_t, preds))
        f3 = open('metrics_gpdd'+str(self.seed)+'.pkl','wb')
        pickle.dump(gp_results,f3)
        #best = comb[getBestParam(gp_results)]
        #self.best_params = best
        return
    
def getBestParam(results):
    best = 0
    for key in results.keys():
        mean_best = np.mean(results[best])
        mean_curr = np.mean(results[key])
        
        if mean_best + np.std(results[best]) > mean_curr + np.std(results[key]):
            if np.sum(results[key] < mean_best)/len(results[key]) >= 0.8:
                best = key
    return best