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
import matplotlib.pyplot as plt

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

selections = ['nested_tournament', 'ranking','double_tournament','roulette','semantic_tournament', 'destabilization_tournament']
#for sel in selections:
#    print(sel)
est_gp = SymbolicRegressor(verbose=1, random_state=0, generations = 100, p_gs_mutation = 0,
                           p_gs_crossover = 0.0,
                           p_point_mutation = 0,
                           p_hoist_mutation = 0,
                           population_size = 100,
                           p_crossover = 0.7,
                           p_subtree_mutation = 0.3,
                           depth_probs = False,
                           hamming_initialization = False,val_set=0.2)

est_gp.fit(fe.training.drop(target_var, axis = 1), fe.training[target_var])


preds = est_gp.predict(fe.unseen.drop(target_var, axis = 1))

mean_squared_error(fe.unseen[target_var], preds)
mean_absolute_error(fe.unseen[target_var], preds)
#Pheno Entropy

pe, = plt.plot(range(1,len(est_gp.recorder.pheno_entropy)), np.array(est_gp.recorder.pheno_entropy[1:])*-1)
#Average Fitness
af, = plt.plot(range(1,len(est_gp.recorder.avgFitness)), np.log(est_gp.recorder.avgFitness[1:]))
#Pheno Variance
pv, = plt.plot(range(1,len(est_gp.recorder.pheno_variance)), np.log(est_gp.recorder.pheno_variance[1:]))
#Depth
d, = plt.plot( range(1,len(est_gp.recorder.depth)), est_gp.recorder.depth[1:])
#Complexity
c, = plt.plot( range(1,len(est_gp.recorder.complexity)), np.log(est_gp.recorder.complexity[1:]))
plt.legend([pe,af,pv,d,c],['Pheno Entropy','Average Fitness','Pheno Variance','Depth','Complexity'])
plt.show()