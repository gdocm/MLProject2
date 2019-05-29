import pandas as pd
import numpy as np
from data_loader import Dataset
from data_preprocessing import PreProcessor
from feature_engineering import FeatureEngineer
from gplearn_MLAA.genetic import SymbolicRegressor
from Models import model_runner
import copy
from Stage import Stage
from CrossProcessor import CrossProcessor
from Processor import Processor
from Toolbox.DataExploration.missing_values import missing_value_reporter, impute_missing_values
from Toolbox.DataExploration.OutlierExploration import z_score_outlier_detection, mahalanobis_distance_outlier
from Toolbox.DataExploration.OutlierTreatment import uni_iqr_outlier_smoothing, drop_outliers
from Toolbox.DataExploration.FeatureSelection import recursive_feature_elimination, pca_extraction

import pickle
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
missingValues.addMethod(missing_value_reporter, ['drop'])

outlierExploration = Stage('Outlier Exploration')
outlierExploration.addMethod(z_score_outlier_detection, [0.03,drop_outliers])
outlierExploration.addMethod(z_score_outlier_detection, [0.03,uni_iqr_outlier_smoothing])

featureSelection = Stage('Feature Selection')
featureSelection.addMethod(recursive_feature_elimination, [8])
featureSelection.addMethod(recursive_feature_elimination, [len(X_train.columns)])

results = []

for seed in range(5):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)
    
    training = X_train.copy()
    training[target_var] = y_train.copy()
    testing = X_test.copy()
    testing[target_var] = y_test.copy()
    
    data = Dataset(training,testing,target_var)
    
    cxP = CrossProcessor(data.training.copy(), data.testing.copy(), target_var, random_state = seed, verbose = True)
    cxP.addStages([missingValues,outlierExploration, featureSelection])
    cxP.mixer()
    
    combination_results = []
    for ix,combination in enumerate(cxP.combinations):
        
        processor = cxP.process()
        print(processor.comb)
        res = model_runner(processor.training.copy(),target_var, processor.testing.copy(),seed)
        results_t = res.scoresDict
        
        filename= open('results_t'+ str(seed)+str(ix)+'.pkl','wb')
        pickle.dump(results_t,filename)
        
        if type(combination_results) == dict:
            for key in combination_results.keys():
                combination_results[key].append(results_t[key])
        else:
            combination_results = {}
            for key in results_t.keys():
                combination_results[key] = [results_t[key]]
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> REST",combination_results)
    results.append(combination_results)
    
#Calulate the average of all seeds

#save = copy.deepcopy(results)

#results_f = {key:np.array(results[0][key]) for key in results[0].keys()}
#for seed in range(1,len(results)):
    for key in results[seed].keys():
        results_f[key]+=np.array(results[seed][key])
#results_f = {key:results_f[key]/5 for key in results_f.keys()}

#final_results ={key:[] for key in results[0].keys()}
#For each combination
#for i in range(len(cxP.combinations)):
    #for each seed
#    for j in range(5):
#        f = open('results_t'+str(j)+str(i)+'.pkl','rb')
#        r = pickle.load(f)
#        for key in r.keys():
#            final_results[key].append(r[key])
    