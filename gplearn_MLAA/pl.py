# -*- coding: utf-8 -*-
"""
Created on Sun May 26 13:30:39 2019

@author: Guilherme
"""

import pickle

filename = open('test.pkl', 'wb')
pickle.dump([1,2,3], filename)

filename = open('test.pkl','rb')

pickle.load(filename)