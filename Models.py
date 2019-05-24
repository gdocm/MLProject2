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
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
import copy

### MODELS
class model_runner():
        
    def __init__(self,training,labels,seed,metric = 'neg_mean_squared_error',cv = 5):
        
        self.training = training
        self.labels = labels
        self.seed = seed
        self.metric = metric
        self.cv = cv
        
        #Models
        regr = RandomForestRegressor(random_state=self.seed)
        bagg = BaggingRegressor(random_state = self.seed)
        ada = AdaBoostRegressor(random_state = self.seed)
        clf = GradientBoostingRegressor(random_state = self.seed)
        est_gp = SymbolicRegressor(random_state=self.seed)
        
        models= [regr,bagg,ada,clf,est_gp]
        modelsStr= ['regr','bagg','ada','clf','est_gp']
        
        #Parameter Grids
        param_grids = []
        
        #Random Forest
        
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start = 20, stop = 50, num = 10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
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
        n_estimators = [int(x) for x in np.linspace(start = 20, stop = 50, num = 10)]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        param_grids.append({name+'__n_estimators': n_estimators,
                        name+'__bootstrap': bootstrap})
    
        #Adaptive Boosting
        name = modelsStr[2]
        n_estimators = [int(x) for x in np.linspace(start = 20, stop = 50, num = 10)]
        learning_rate = [0.1,0.3,0.6,0.9]
        # Create the random grid
        param_grids.append({name+'__n_estimators': n_estimators,
                        name+'__learning_rate': learning_rate})
    
        #Gradient Boosting

        n_estimators = [int(x) for x in np.linspace(start = 20, stop = 50, num = 10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        learning_rate = [0.1,0.3,0.6,0.9]
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
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
    
        #Genetic Programming
        population_size=[20,40,60]
        generations=[20,40,100],
        p_crossover=[0.1,0.3,0.6,0.9]
        p_subtree=[0.9,0.6,0.3,0.1]  
        name= modelsStr[4]
        param_grids.append({name+'__population_size': population_size,
               name+'__generations':generations,
               name+'__p_crossover': p_crossover,
               name+'__p_subtree_mutation': p_subtree,
               name+'__p_gs_mutation':[0],
               name+'__p_gs_crossover':[0],
               name+'__p_hoist_mutation':[0],
               name+'__p_point_mutation': [0]})

            
        #Hyper Parameter Optimization
        #Grid_Search
        #for m in range(len(models)):
        m = 4
        models[m] = self._gridSearchModel(modelsStr[m], models[m], param_grids[m])
        
        scores = []
        kf = KFold(5)
        for model in models:
            model_scores= []
            for train_index, test_index in kf.split(self.training):
                model.fit(self.training.iloc[train_index],self.labels[train_index])
                preds = model.predict(self.training.iloc[test_index])
                model_scores.append(mean_squared_error(self.labels.iloc[test_index], preds))
            model_scores = np.mean(model_scores)
            scores.append(model_scores)
        self.scoresDict = dict(zip(modelsStr, scores))
        
        
        
    def _gridSearchModel(self,model_name, model, param_grid, cv = 5):
        print(">>>>>>>>>>>> Optimizing " + model_name)
        pipeline = Pipeline([(model_name, model)])
        print(self.training.dtypes)
        print(self.labels.dtypes)
        estimator = GridSearchCV(pipeline, param_grid, cv=cv, n_jobs=-1,verbose = 1,scoring=self.metric)
        estimator.fit(self.training, self.labels)
        return estimator