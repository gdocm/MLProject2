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
outlierExploration.addMethod(mahalanobis_distance_outlier, [drop_outliers])
outlierExploration.addMethod(z_score_outlier_detection, [drop_outliers])
outlierExploration.addMethod(z_score_outlier_detection, [uni_iqr_outlier_smoothing])

featureSelection = Stage('Feature Selection')
featureSelection.addMethod(recursive_feature_elimination, [5])
featureSelection.addMethod(pca_extraction, [0.8])



results = []
test_results = []
for seed in range(2):
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
    results_combination = []
    for combination in cxP.combinations:
        
        processor = cxP.process()
        res = model_runner(processor.training,target_var, processor.testing,seed)
        results_t = res.scoresDict
        test_results_t = res.tscoresDict
        if type(results) == dict:
            for key in results.keys():
                results[key] = [results[key]]
                results[key].append(results_t[key])
        else:
            results = results_t
            
        if type(test_results) == dict:
            for key in results.keys():
                results[key] = [results[key]]
                results[key].append(results_t[key])
        else:
            test_results = test_results
        results_combination.append((results,test_results))
        
#Calulate the average of all seeds

save = copy.deepcopy(results)
save2 = copy.deepcopy(test_results)
for c in results_combination:
    c[0]= {k: (np.mean(v), np.std(v)) for k,v in c[0].items()}
    c[1]= {k: (np.mean(v), np.std(v)) for k,v in c[1].items()}