# -*- coding: utf-8 -*-
"""
Created on Sat May 18 17:14:52 2019

@author: Guilherme
"""
import numpy as np
from sklearn.impute import SimpleImputer

### Missing Values

def missing_value_reporter(dataframe, threshold = 0.03, report = True):
    
    '''
    Reports on percentages of missing values per columns 
    returning columns above threshold
    '''

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
    return columns


def convert_numeric_labelling(self,var):
    temp = self.training[var].dropna().copy()
    unique = temp.drop_duplicates()
    var_dict = {}
    for ix,value in enumerate(unique):
        var_dict[value] = ix
    self.training[var] = self.training[var].apply(lambda x: var_dict[x] if x in var_dict.keys() else None)
    return var_dict

def revert_numeric_labelling(self,var,var_dict):
    cat_dict = {}
    for k,v in var_dict.items():
        cat_dict[v] = k
    self.training[var] = self.training[var].apply(lambda x: cat_dict[x] if x in cat_dict.keys() else None)

def _impute_missing_values(self):
    self.report.append('_impute_missing_values')
    for column in self.training[self.numerical_var]:
        data = self.training[column]
        loc, scale = norm.fit(data)
        n = norm(loc = loc, scale = scale)
        _, p_value = kstest(self.training[column].values, n.cdf)
        if p_value > 0.05:
            self._imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            self.training[column] = self._imputer.fit_transform(self.training[column].values.reshape(-1,1))
            self.unseen[column] = self._imputer.transform(self.unseen[column].values.reshape(-1,1))
        else:
            self._imputer = SimpleImputer(missing_values=np.nan, strategy='median')
            self.training[column] = self._imputer.fit_transform(self.training[column].values.reshape(-1,1))
            self.unseen[column] = self._imputer.transform(self.unseen[column].values.reshape(-1,1))
    for var in self.cat_vars:

        var_dict = self.convert_numeric_labelling(var)
        self.training[var] = self.training[var].astype(int)
        self.unseen[var] = self.unseen[var].astype(int)
        self._imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        self.training[var] = self._imputer.fit_transform(self.training[var].values.reshape(-1,1))
        self.unseen[var] = self._imputer.transform(self.unseen[var].values.reshape(-1,1))

        self.training[var] = self.training[var].astype('category')
        self.unseen[var] = self.unseen[var].astype('category')

        self.revert_numeric_labelling(var,var_dict)
    self.training[self.cat_vars] =self.training[self.cat_vars].astype('category')