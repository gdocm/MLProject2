# -*- coding: utf-8 -*-
"""
Created on Wed May 22 19:17:56 2019

@author: Guilherme
"""
import numpy as np

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
        
    def compute_metrics(self, X):
        self.pheno_entropy.append(self.phenoEntropy())
        self.pheno_variance.append(self.phenoVariance())
        self.depth.append(self.avgDepth())
        self.length.append(self.avgLength())
        self.avgFitness.append(np.mean(self.fitness))
        #self.complexity.append(self.computeComplexity(X))
                
    def phenoEntropy(self):
        unique_fitnesses, counts = np.unique(self.fitness, return_counts = True)
        return np.sum([counts[fitness]*np.log(counts[fitness]) for fitness in range(len(unique_fitnesses))])

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
    
    def computeComplexity(self,X):

        return np.mean(self.computeProgramComplexity(program,X) for program in self.population)
            
    def computeProgramComplexity(self, program, X):
        
        return np.sum([self.computePartialComplexity(program,X,j) for column in X.shape[1]])/X.shape[0]
            
    def computePartialComplexity(self, program, X, column):
        semantics = self.execute()
        p = X[column]
        q = p.sort_values()
        
        sum_ = 0
        for value in range(len(q)-2):
            temp1 = (semantics[value + 1] - semantics[value])/(q[value+1]-q[value])
            temp2 = (semantics[value + 2] - semantics[value+1])/(q[value+2]-q[value+1])
            sum_ += np.abs(temp1 - temp2)
        return sum_
        
        