# -*- coding: utf-8 -*-
"""
Created on Fri May 31 15:12:38 2019

@author: Guilherme
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import pandas as pd
import pandas as pd
import numpy as np
from data_loader import Dataset
from data_preprocessing import PreProcessor
from feature_engineering import FeatureEngineer
from gplearn_MLAA.genetic import SymbolicRegressor
from Models2 import model_runner
import copy
from Stage import Stage
from CrossProcessor import CrossProcessor
from Processor import Processor
from Toolbox.DataExploration.missing_values import missing_value_reporter, impute_missing_values
from Toolbox.DataExploration.OutlierExploration import z_score_outlier_detection, mahalanobis_distance_outlier
from Toolbox.DataExploration.OutlierTreatment import uni_iqr_outlier_smoothing, drop_outliers
from Toolbox.DataExploration.FeatureSelection import recursive_feature_elimination, pca_extraction
from sklearn.model_selection import KFold
    
#Data Loading
data = pd.read_csv('Data\\data.csv')
data.drop('Unnamed: 0', axis  = 1, inplace = True)

entities = dict(zip(data.index, data['Entity']))
data.drop('Entity', axis = 1, inplace = True)

X_train = data.drop('mortality_rate',axis = 1)
y_train = data['mortality_rate']
target_var = 'mortality_rate'

#Preprocessing Stages

missingValues = Stage('Missing Value Treatment')
missingValues.addMethod(missing_value_reporter, ['impute'])

outlierExploration = Stage('Outlier Exploration')
outlierExploration.addMethod(z_score_outlier_detection, [0.03,uni_iqr_outlier_smoothing])

featureSelection = Stage('Feature Selection')
featureSelection.addMethod(recursive_feature_elimination, [len(X_train.columns)])

processor = Processor(None,None,None,0)
processor.addStages([missingValues,outlierExploration,featureSelection])

results = []
params = []
for seed in range(5):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)
    
    training = X_train.copy()
    training[target_var] = y_train.copy()
    testing = X_test.copy()
    testing[target_var] = y_test.copy()
    
    data = Dataset(training,testing,target_var)
    processor.training = data.training
    processor.testing = data.testing
    processor.target_var = target_var
    processor.seed = seed
    processor.exec_()

    model = SymbolicRegressor(verbose=1, random_state = 0, generations = 100, p_gs_mutation = 0,
                               p_gs_crossover = 0.0,
                               p_point_mutation = 0,
                               p_hoist_mutation = 0,
                               population_size = 100,
                               p_crossover = 0.1,
                               p_subtree_mutation = 0.9,
                               depth_probs = False,
                               hamming_initialization = False,val_set=0.2)
    
    model.fit(processor.training.drop(target_var, axis = 1), processor.training[target_var])


    preds = model.predict(processor.testing.drop(target_var, axis = 1))


mean_absolute_error(processor.testing[target_var], preds)
#Pheno Entropy
pe, = plt.plot(range(1,len(model.recorder.pheno_entropy)), np.array(model.recorder.pheno_entropy[1:])*-1)
#Average Fitness
af, = plt.plot(range(1,len(model.recorder.avgFitness)), np.log(model.recorder.avgFitness[1:]))
#Average Val Fitness
avf, = plt.plot(range(1,len(model.recorder.avgValFitness)), np.log(model.recorder.avgValFitness[1:]))
#Difference
diff = np.array(model.recorder.avgFitness) - np.array(model.recorder.avgValFitness)
di, = plt.plot(range(1,len(diff)), diff[1:])
#Pheno Variance
pv, = plt.plot(range(1,len(model.recorder.pheno_variance)), np.log(model.recorder.pheno_variance[1:]))
#Depth
d, = plt.plot( range(1,len(model.recorder.depth)), model.recorder.depth[1:])
#Complexity
c, = plt.plot( range(1,len(model.recorder.complexity)), np.log(model.recorder.complexity[1:]))
plt.legend([pe,af,pv,d,c],['Pheno Entropy','Average Fitness','Pheno Variance','Depth','Complexity'])
plt.show()