# -*- coding: utf-8 -*-
"""
Created on Sat May 18 17:14:52 2019

@author: Guilherme
"""
import numpy as np
from sklearn.impute import SimpleImputer
from scipy.stats import norm, kstest

### Missing Values
def missing_value_reporter(self,  method='impute', threshold = 0.03, report = True):
    
    '''
    Reports on percentages of missing values per columns 
    returning columns above threshold
    '''
    dataframe = self.training.copy()
    columns = []
    for column in dataframe.columns:
        percentage = round(np.sum(dataframe[column].isna())/dataframe.shape[0],2)
        percentage = percentage*100
        if percentage > threshold*100 and report:       
            print(column+': ' + ' ' + str(percentage) + '%')
        if percentage > threshold*100:
            columns.append(column)
    total_rows = dataframe[columns].shape[0] - dataframe[columns].dropna().shape[0]
    percent = total_rows/dataframe.shape[0]
    if report:
        print('\n Total Removed Rows (' + str(threshold) + '): ' + str(total_rows) + ' ' + str(round(percent*100,2)) +'%')
    if method == 'impute':
        print("Impute")
        return impute_missing_values(self)
        
    
    elif method == 'drop':
        self.training.dropna(inplace = True)
        self.testing.dropna(inplace = True)
        print("Drop M")
        return self


def convert_numeric_labelling(dataframe,var):
    temp = dataframe[var].dropna().copy()
    unique = temp.drop_duplicates()
    var_dict = {}
    for ix,value in enumerate(unique):
        var_dict[value] = ix
    dataframe[var] = dataframe[var].apply(lambda x: var_dict[x] if x in var_dict.keys() else None)
    return var_dict

def revert_numeric_labelling(dataframe,var,var_dict):
    cat_dict = {}
    for k,v in var_dict.items():
        cat_dict[v] = k
    dataframe[var] = dataframe[var].apply(lambda x: cat_dict[x] if x in cat_dict.keys() else None)

def impute_missing_values(self):
    
    for column in self.numerical_vars:
        data = self.training[column]
        loc, scale = norm.fit(data)
        n = norm(loc = loc, scale = scale)
        _, p_value = kstest(self.training[column].values, n.cdf)
        if p_value > 0.05:
            _imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            self.training[column] = _imputer.fit_transform(self.training[column].values.reshape(-1,1))
            self.testing[column] = _imputer.transform(self.testing[column].values.reshape(-1,1))
        else:
            _imputer = SimpleImputer(missing_values=np.nan, strategy='median')
            self.training[column] = _imputer.fit_transform(self.training[column].values.reshape(-1,1))
            self.testing[column] = _imputer.transform(self.testing[column].values.reshape(-1,1))
    if len(self.categorical_vars) > 0:
        for var in self.categorical_vars:
            var_dict = convert_numeric_labelling(var)
            self.training[var] = self.training[var].astype(int)
            self.testing[var] = self.testing[var].astype(int)
            _imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
            self.training[var] = _imputer.fit_transform(self.training[var].values.reshape(-1,1))
            self.testing[var] = _imputer.transform(self.testing[var].values.reshape(-1,1))
    
            self.training[var] = self.training[var].astype('category')
            self.testing[var] = self.testing[var].astype('category')
    
            revert_numeric_labelling(self.training,var,var_dict)
        
        self.training[self.categorical_vars] = self.training[self.categorical_vars].astype('category')
    return self