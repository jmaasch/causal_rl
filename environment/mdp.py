import numpy as np
import tqdm 
import random

class MDP:
    #kernelx [x_{t-1},y_t,a] denotes P(x_t|x_{t-1},y_t,a)
    #kernely [x_{t-1},y_{t-1},a] denotes P(y_t|x_{t-1},y_{t-1},a)
    def __init__(self,kernelx,kernely,reward):
        self.kernelx=kernelx
        self.kernely=kernely
        self.reward=reward
        self.n_x,self.n_y,self.n_action,_=kernelx.shape
    #generate the next state given current state and action
    def sample(self,s,a):
        s_=np.zeros(2)
        x=s[0]
        y=s[1]
        #first generate the next y
        proby=self.kernely[int(x),int(y),a]
        y_=np.random.choice(np.arange(self.n_y),p=proby)
        #generate next x
        probx=self.kernelx[int(x),int(y_),a]
        x_=np.random.choice(np.arange(self.n_x),p=probx)
        s_[0]=x_
        s_[1]=y_

        return(s_)
    #generate the reward given current state and action
    def rewards(self,s,a):
        return self.reward[int(s[0]),int(a)]
    #generate the whole sample path
    def sample_path(self,s_init,pi,length):
        path=[]
        s=s_init
        for n in tqdm.tqdm(range(length)):
            a      = pi.play(s)
            s_ = self.sample(s, a)
            rew=self.rewards(s,a)
            path.append((s, a, rew, s_))
            s=s_
        return path

