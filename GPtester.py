# -*- coding: utf-8 -*-

"""
Created on Wed May 22 18:45:07 2019
@author: Guilherme
"""
import pandas as pd
from data_loader import Dataset
from data_preprocessing import PreProcessor
from feature_engineering import FeatureEngineer
from gplearn_MLAA.genetic import SymbolicRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from gplearn_MLAA.Recorder import Recorder
import seaborn as sb
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
import numpy as np

data = pd.read_csv('Data//data.csv')
data.drop('Unnamed: 0', axis  = 1, inplace = True)

entities = dict(zip(data.index, data['Entity']))
data.drop('Entity', axis = 1, inplace = True)

X_train = data.drop('mortality_rate',axis = 1)
y_train = data['mortality_rate']
target_var = 'mortality_rate'

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

training = X_train.copy()
training[target_var] = y_train.copy()
testing = X_test.copy()
testing[target_var] = y_test.copy()

data = Dataset(training,testing,target_var)
pr = PreProcessor(data.training, data.testing, target_var, 0)
fe = FeatureEngineer(pr.training, pr.unseen, target_var, 0)
results = []


est_gp = SymbolicRegressor(verbose=1, random_state=0, generations = 200, p_gs_mutation = 0,
                           p_gs_crossover = 0,
                           p_point_mutation = 0,
                           p_hoist_mutation = 0,
                           population_size = 1000,
                           p_crossover = 0.1,
                           p_subtree_mutation = 0.1,
                           p_grasm_mutation = 0.9,
                           dynamic_depth = True,
                           depth_probs = True,
                           hue_initialization_params=True)

est_gp.fit(fe.training.drop(target_var, axis = 1), fe.training[target_var])

selections = ['nested_tournament', 'ranking','double_tournament','roulette','semantic_tournament']
for sel in selections:
    print(sel)
    est_gp = SymbolicRegressor(verbose=1, random_state=0, generations = 20, p_gs_mutation = 0,
                               p_gs_crossover = 0.0,
                               p_point_mutation = 0,
                               p_hoist_mutation = 0,
                               population_size = 30,
                               p_crossover = 0.7,
                               p_subtree_mutation = 0.3,
                               depth_probs = False,
                               selection = sel)
    
    est_gp.fit(fe.training.drop(target_var, axis = 1), fe.training[target_var])


preds = est_gp.predict(fe.unseen.drop(target_var, axis = 1))

mean_squared_error(fe.unseen[target_var], preds)
mean_absolute_error(fe.unseen[target_var], preds)

sb.lineplot(x = range(0,len(est_gp.recorder.pheno_entropy)), y = est_gp.recorder.pheno_entropy)
sb.lineplot(x = range(0,len(est_gp.recorder.avgFitness)), y = np.log(est_gp.recorder.avgFitness))
sb.lineplot(x = range(0,len(est_gp.recorder.pheno_variance)), y = np.log(est_gp.recorder.pheno_variance))
sb.lineplot(x = range(0,len(est_gp.recorder.depth)), y = est_gp.recorder.depth)
