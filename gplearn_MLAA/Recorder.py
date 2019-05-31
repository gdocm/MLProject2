# -*- coding: utf-8 -*-
"""
Created on Wed May 22 19:17:56 2019

@author: Guilherme
"""
import numpy as np
import pandas as pd
import pickle
import math

class Recorder:
    
    def __init__(self, generations):
        self.generations = generations
        self.avgFitness = []
        self.gen_entropy = []
        self.pheno_entropy = []
        self.gen_variance = []
        self.pheno_variance = []
        self.fitness = None
        self.population = None
        self.depth = []
        self.length = []
        self.complexity = []
        self.gen = 0
        
    def compute_metrics(self, X):
        self.pheno_entropy.append(self.phenoEntropy())
        self.pheno_variance.append(self.phenoVariance())
        self.depth.append(self.avgDepth())
        self.length.append(self.avgLength())
        self.avgFitness.append(np.mean(self.fitness))
        #self.savepopulation()
        #self.complexity.append(self.computeComplexity(X))
    
    def savepopulation(self):
        filename = open('pop'+str(self.gen)+'.pkl','wb')
        pickle.dump(self.population, filename)
        self.gen += 1
    
    def phenoEntropy(self):
        unique_fitnesses, counts = np.unique(self.fitness, return_counts = True)
        return -np.sum([counts[fitness]*np.log(counts[fitness]) for fitness in range(len(unique_fitnesses))])

    def genoEntropy(self, pop):
        return np.sum([nmr_Ind_with_struct*np.log(nmr_Ind_with_struct) for struct in unique_structrs])
    
    def phenoVariance(self):
        return np.sum([(fitness -np.mean(self.fitness))**2 for fitness in self.fitness])/(len(self.fitness)-1)
    
    def genoVariance(self, pop):
        return np.sum([(distance_o -avg(distances_o))**2 for distance_o in distances_o])/(len(distances_o)-1)
        
    def avgDepth(self):
        return np.mean([program._depth() for program in self.population])
    
    def avgLength(self):
        return np.mean([program._length() for program in self.population])
    
    def ccomplex(self,X):
        for i in range(self.generations):
            filename = open('pop'+str(i)+'.pkl','rb')
            self.population = pickle.load(filename)
            self.complexity.append(self.computeComplexity(X))
            
    
    def computeComplexity(self,X):
        
        return np.mean([self.computeProgramComplexity(program,X) for program in self.population])
            
    def computeProgramComplexity(self, program, X):
        
        return np.sum([self.computePartialComplexity(program,X,j) for j in range(X.shape[1])])/X.shape[1]
            
    def computePartialComplexity(self, program, X, column):
        semantics = program.execute(X)
        p_ = pd.DataFrame(X)[column]
        q_ = p_.sort_values()
        
        sum_ = 0
        for value in range(len(q_)-2):
            temp1 = (semantics[value + 1] - semantics[value])/(q_[value+1]-q_[value])
            temp2 = (semantics[value + 2] - semantics[value+1])/(q_[value+2]-q_[value+1])
            r = np.abs(temp1 - temp2)
            if r == np.inf or math.isnan(r):
                r = 0.0000000000001
            
            sum_ += r
        if math.isnan(sum_) or sum_ == np.inf:
            raise Exception(sum_)
        return sum_
        
        