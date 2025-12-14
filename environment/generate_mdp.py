from numpy.random import SeedSequence,default_rng,Generator
import numpy as np
class Generate_MDP:
    def __init__(self,n_statex,n_statey,n_action):
        self.n_statex=n_statex
        self.n_statey=n_statey
        self.n_action=n_action
    def generate_x(self,random:Generator):
        kernel_xa=default_rng(random).random(size=(self.n_statex,self.n_statey,self.n_statex))
        kernel_xa/=kernel_xa.sum(axis=-1, keepdims=True)
        kernel_x=np.broadcast_to(kernel_xa[:, :, None, :], (self.n_statex, self.n_statey, self.n_action, self.n_statex)).copy()
        return kernel_x
    def generate_y(self,random:Generator):
        kernel_y=np.empty((self.n_statex,self.n_statey,self.n_action,self.n_statey),dtype=float)
        kernel_0=default_rng(random).random(size=(self.n_statex,self.n_statey))
        kernel_0/=kernel_0.sum(axis=-1,keepdims=True)
        kernel_y[:,:,0,:]=kernel_0[:,None,:]
        kernel_1=default_rng(random).random(size=(self.n_statex,self.n_statey,self.n_statey))
        kernel_1/=kernel_1.sum(axis=-1,keepdims=True)
        kernel_y[:,:,1,:]=kernel_1
        return kernel_y
    def generate_reward(self,random:Generator):
        return default_rng(random).random(size=(self.n_statex,self.n_action))