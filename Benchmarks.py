# -*- coding: utf-8 -*-
"""
Created on Thu May 30 18:18:53 2019

@author: Guilherme
"""
import pickle
import numpy as np
import seaborn as sb
import pandas as pd
import pandas as pd
import numpy as np
from data_loader import Dataset
from data_preprocessing import PreProcessor
from feature_engineering import FeatureEngineer
from gplearn_MLAA.genetic import SymbolicRegressor
from ModelsInit import model_runner
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
    
    res = model_runner(processor.training.copy(),target_var, processor.testing.copy(),seed)
    params.append(res.best_params)
    results.append(res.scoreDict)


f = open('results_gpinit.pkl','wb')
pickle.dump(results,f)

f2 = open('params_gpinit.pkl','wb')
pickle.dump(params,f2)