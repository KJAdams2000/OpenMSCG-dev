#!/user/bin/env python3

import numpy as np

class WeightingLD:
    
    def __init__(self, **kwargs):
        self.I      = "undefined"
        self.J      = "undefined"
        self.low    = "undefined"
        self.high   = "undefined"
        return
    
    def init(self):
        pass
    
    def compute(self, top, traj, weights):
        tid = top.names_atom.index(self.I)
        types = top.types_atom
        
        lmd = traj.l    
        
        for i in range(top.n_atom):
            if types[i] == tid:
                weights[i] = [(self.high, lmd[i]), (self.low, 1.0-lmd[i])]
