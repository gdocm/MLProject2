# -*- coding: utf-8 -*-
"""
Created on Thu May 30 18:18:53 2019

@author: Guilherme
"""
import pickle
import numpy as np
import seaborn as sb
import pandas as pd
#Benchmarks
modelsStr= ['regr','bagg','ada','clf','est_gp']

final_results =[]
#For each combination
for i in range(2):
    #for each seed
    combination_results = {key:[] for key in modelsStr}
    for j in range(5):
        f = open('results_t'+str(j)+str(i)+'.pkl','rb')
        r = pickle.load(f)
        for key in r.keys():
            combination_results[key].append(r[key])
    combination_results = {key:(np.mean(combination_results[key]),np.std(combination_results[key])) for key in combination_results.keys()}
    final_results.append(combination_results)

fr = {key:[] for key in modelsStr}