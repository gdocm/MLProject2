# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 19:11:05 2019

@author: Guilherme
"""
import numpy as np
import pickle
import pandas as pd
import seaborn as sb
import copy
results = []
for i in range(5):
    f = open('metrics_gpmut'+str(i) + '.pkl','rb')
    results.append(pickle.load(f))

f5= open('sem_mut.pkl','rb')
results2 = pickle.load(f5)

#Get Mean of Models per cv
models = {key:[] for key in results[0].keys()}
for seed in range(len(results)):
    for model in range(len(results[seed])):
        models[model].append(np.mean(results[seed][model]))

#Get Mean of Models per seed
fr = []
for key in models.keys():
    t = pd.DataFrame()
    t['mae'] = models[key]
    t['mode'] =[key]*5
        
    fr.append(t)

cdf = pd.concat(fr)  
cdf.columns = ['Mean Absolute Error','Model']
sb.boxplot(data =cdf ,y='Mean Absolute Error', x='Model')

final
def getBestParam(results):
    best = 0
    for key in results.keys():
        mean_best = np.mean(results[best])
        mean_curr = np.mean(results[key])
        
        if mean_best + np.std(results[best]) > mean_curr + np.std(results[key]):
            if np.sum(results[key] < mean_best)/len(results[key]) >= 0.8:
                best = key
    return best