# -*- coding: utf-8 -*-
"""
Created on Sun May 19 14:50:08 2019

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
        
        #Models
        regr = RandomForestRegressor(random_state=self.seed)
        bagg = BaggingRegressor(random_state = self.seed)
        ada = AdaBoostRegressor(random_state = self.seed)
        clf = GradientBoostingRegressor(random_state = self.seed)
        
        
        models= [regr,bagg,ada,clf]
        modelsStr= ['regr','bagg','ada','clf']
        
        #Parameter Grids
        param_grids = []
        
        #Random Forest
        
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start = 20, stop = 50, num = 4)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num = 4)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        name= modelsStr[0]
        param_grids.append({name +'__n_estimators': n_estimators,
                       name+'__max_features': max_features,
                       name+'__max_depth': max_depth,
                       name+'__min_samples_split': min_samples_split,
                       name + '__min_samples_leaf': min_samples_leaf,
                       name+'__bootstrap': bootstrap})
        
        
        #Bagging Regressor
        name= modelsStr[1]
        n_estimators = [int(x) for x in np.linspace(start = 20, stop = 50, num = 4)]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        param_grids.append({name+'__n_estimators': n_estimators,
                        name+'__bootstrap': bootstrap})
    
        #Adaptive Boosting
        name = modelsStr[2]
        n_estimators = [int(x) for x in np.linspace(start = 20, stop = 50, num = 4)]
        learning_rate = [0.1,0.5,0.9]
        # Create the random grid
        param_grids.append({name+'__n_estimators': n_estimators,
                        name+'__learning_rate': learning_rate})
    
        #Gradient Boosting

        n_estimators = [int(x) for x in np.linspace(start = 20, stop = 50, num = 4)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        learning_rate = [0.1,0.5,0.9]
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num = 4)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        name= modelsStr[3]
        param_grids.append({name+'__n_estimators': n_estimators,
                       name+'__learning_rate':learning_rate,
                       name+'__max_features': max_features,
                       name+'__max_depth': max_depth,
                       name+'__min_samples_split': min_samples_split,
                       name+'__min_samples_leaf': min_samples_leaf})

            
        #Hyper Parameter Optimization
        #Grid_Search
        for m in range(len(models)):
            models[m] = self._gridSearchModel(modelsStr[m], models[m], param_grids[m])
        modelsStr.append('est_gp')
        p_crossover=[0.1,0.5,0.9]
        p_subtree=[0.9,0.5,0.1]  
        rs = [self.seed] * 3
        param_grid_gp = {
               'p_crossover': p_crossover,
               'p_subtree_mutation': p_subtree,
               'random_state':rs}
        
        models.append(self.gridSearchGp(param_grid_gp))
        scores = []
        for model in models:
            preds = models[2].predict(self.testing)
        scores.append(mean_absolute_error(self.labels_t, preds))
        self.scoresDict = dict(zip(modelsStr[2], scores))
        
        
    def _gridSearchModel(self,model_name, model, param_grid, cv = 5):
        print(">>>>>>>>>>>> Optimizing " + model_name)
        pipeline = Pipeline([(model_name, model)])
        estimator = GridSearchCV(pipeline, param_grid, cv=cv, n_jobs=-1,verbose = 1,scoring=self.metric)
        estimator.fit(self.training, self.labels)
        return estimator
    
    def gridSearchGp(self,param_grid):
        
        parameters = list(param_grid.values())
        comb = []
        for i in range(len(parameters[0])):
            t =  {}
            for j in param_grid.keys():
                t[j] = param_grid[j][i]
            comb.append(t)
        kf = KFold(2)
        gp_results = {}
        for c in range(len(comb)):
            gp_results[c] = []
        for train_index, test_index in kf.split(self.training):
            for c in range(len(comb)):
                est_gp = SymbolicRegressor(**comb[c])
                est_gp.fit(self.training.iloc[train_index], self.labels.iloc[train_index])
                preds = est_gp.predict(self.training.iloc[test_index])
                
                gp_results[c] = mean_absolute_error(self.labels.iloc[test_index], preds)
        best = comb[np.argmin([np.mean(gp_results[key]) for key in gp_results.keys()])]
        estimator = SymbolicRegressor(**best)
        estimator.fit(self.training,self.labels)
        return estimator
        