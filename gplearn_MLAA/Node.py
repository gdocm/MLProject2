# -*- coding: utf-8 -*-
"""
Created on Fri May 24 18:00:38 2019

@author: Guilherme
"""

from .functions import _Function, _function_map, sig1, tanh1

class Node:
    def __init__(self, node, depth = 1, children = []):
        self.node = node
        self.children = children
        self.depth = depth
                
    def setDepth(self):
        for child in self.children:
            if type(child) == Node:
                child.depth += 1
                child.setDepth()
        