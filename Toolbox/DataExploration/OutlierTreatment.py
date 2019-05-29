# -*- coding: utf-8 -*-
"""
Created on Tue May 28 12:50:36 2019

@author: Guilherme
"""

#Outlier Treatment
import numpy as np
from scipy.stats import zscore,iqr

def uni_iqr_outlier_smoothing(self,DOAGO_results):
    '''use the info gatheres from the previous function (decide on and get outliers) and smoothes the detected outliers.'''
    ds = self.training
    novo_ds = ds.copy()
    for key in DOAGO_results.keys():
        for var in DOAGO_results[key]:
            if var != 'multi/unknown':
                if ds[var][ds.index == key].values > np.mean(ds[var]):
                    novo_ds[var][novo_ds.index == key] = np.percentile(ds[var], 75) + 1.5 * iqr(ds[var])

                else:
                    novo_ds[var][novo_ds.index == key] = np.percentile(ds[var], 25) - 1.5 * iqr(ds[var])
            else: novo_ds=novo_ds[novo_ds.index!=key]
    self.training = novo_ds
    print("SMMOTH")
    return self

def drop_outliers(self,outliers):
    print("Drop Outliers")
    self.training.drop(index = list(outliers), inplace = True)
    return self
    
