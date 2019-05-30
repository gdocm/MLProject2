# -*- coding: utf-8 -*-
"""
Created on Thu May 30 18:18:53 2019

@author: Guilherme
"""
import pickle
#Benchmarks
modelsStr= ['regr','bagg','ada','clf','est_gp']

final_results ={key:[] for key in modelsStr}
#For each combination
for i in range(8):
    #for each seed
    for j in range(5):
        f = open('results_t'+str(j)+str(i)+'.pkl','rb')
        r = pickle.load(f)
        for key in r.keys():
            final_results[key].append(r[key])