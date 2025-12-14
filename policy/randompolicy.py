import numpy as np
class Random:
    def __init__(self,model):
        self.n_action=model.n_action
    def play(self,s):
        return np.random.randint(0,self.n_action)
    