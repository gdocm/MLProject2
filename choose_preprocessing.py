#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 11:15:08 2019

@author: rizzoli
"""

import pandas as pd
from data_loader import Dataset
from data_preprocessing import Processor
from feature_engineering import FeatureEngineer
from gplearn_MLAA.genetic import SymbolicRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from gplearn_MLAA.Recorder import Recorder
import seaborn as sb
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.model_selection import KFold
import statsmodels.api as sm
 from sklearn.model_selection import train_test_split

data = pd.read_csv('Data//data.csv')
data.drop('Unnamed: 0', axis  = 1, inplace = True)
data.drop(columns='Entity',inplace=True)

def get_best_proc_battle(data):
    
    seeds=5
    
    X_train = data.drop('mortality_rate',axis = 1)
    y_train = data['mortality_rate']
    target_var = 'mortality_rate'
    
    for seed in range(seeds):   
        reg_erros=[]
        
       
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)
        training = X_train.copy()
        training[target_var] = y_train.copy()
        testing = X_test.copy()
        testing[target_var] = y_test.copy()
        
        kf = KFold(n_splits=5,random_state=seed, shuffle=False)
        
        for train_index, test_index in kf.split(X_train):
            
            x_train, X_val = training.iloc[train_index].values,training.iloc[test_index].values
            Y_train,y_val =training[target_var].iloc[train_index].values,training[target_var].iloc[test_index].values
            
            
            pr=Processor(training.iloc[train_index],training.iloc[test_index],'mortality_rate',seed)

            
            model = sm.OLS(pr.training['mortality_rate'],pr.training.drop(columns='mortality_rate').values).fit()
            predictions = model.predict(pr.unseen.drop(columns='mortality_rate').values)
            reg_erros.append(np.square(np.subtract(pr.unseen['mortality_rate'], predictions)).mean())
    
    return np.mean(reg_erros)
    
        
error_imputing=get_best_proc_battle(data)



