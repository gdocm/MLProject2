import pandas as pd
import numpy as np
from data_loader import Dataset
from data_preprocessing import Processor
from feature_engineering import FeatureEngineer
from gplearn_MLAA.genetic import SymbolicRegressor
from Models import model_runner

data = pd.read_csv('Data\\data.csv')
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
pr = Processor(data.training, data.testing, target_var, 0)
fe = FeatureEngineer(pr.training, pr.unseen, target_var, 0)
results = []

for seed in range(2):
    temp = model_runner(pr.training.drop(target_var, axis = 1),pr.training[target_var], seed).scoresDict
    if type(results) == dict:
        for key in results.keys():
            results[key] = [results[key]]
            results[key].append(temp[key])
    else:
        results = temp

#Calulate the average of all seeds
results = {k:np.mean(v) for k,v in results.items()}

### Entroypy and Variance Tests
#pheno
def pheno_entropy(pop):
    return np.sum([nmr_Ind_with_fitness*np.log(nmr_Ind_with_fitness) for fitness in unique_fitnesses])

def geno_entropy(pop):
    return np.sum([nmr_Ind_with_struct*np.log(nmr_Ind_with_struct) for struct in unique_structrs])

def pheno_variance(pop):
    return np.sum([(fitness -avg(fitnesses))**2 for fitness in fitnesses])/(len(fitnesses)-1)

def geno_variance(pop):
    return np.sum([(distance_o -avg(distances_o))**2 for distance_o in distances_o])/(len(distances_o)-1)
