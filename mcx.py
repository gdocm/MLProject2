# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 19:11:05 2019

@author: Guilherme
"""
import numpy as np
import pickle
results = []
for i in range(1):
    f = open('metrics_gpcx'+str(i) + '.pkl','rb')
    results.append(pickle.load(f))

final = {key:[] for key in results[0].keys()}
for res in results:
    for key in res:
        final[key].append(res[key])

for key in final.keys():
    final[key] = (np.mean(final[key]),np.std(final[key]))

def getBestParam(results):
    best = 0
    for key in results.keys():
        mean_best = np.mean(results[best])
        mean_curr = np.mean(results[key])
        
        if mean_best + np.std(results[best]) > mean_curr + np.std(results[key]):
            if np.sum(results[key] < mean_best)/len(results[key]) >= 0.8:
                best = key
    return best