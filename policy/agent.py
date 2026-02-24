import numpy as np


def iterate_algorithm(model, algorithm, history, rsum,episodes=None):
    """ 
    Iterate a reinforcement learning algorithm.
    """
    x,done,truncated=history[-1]
    if done or truncated:
        x,_=model.reset(seed=rsum)
        # rsum+=r
        # r=rsum/len(history)
        # print(r)



    
    a=algorithm.act(x)
    y,r,done,truncated,_=model.step(a)
    algorithm.observe(x,a,r,y,done,truncated)
    history[-1]=(x,a,r,y,done,truncated)
    history.append((y,done,truncated))


def parse_history(model, history,gamma=np.zeros(1000000),T=100,model_type="discrete"):
    """ Parse the history. Return a dictionary with the following entries:

    average reward: sum r_i/t
   
    
    """

    info = dict()
    info["average reward"] = []
    info["average expected reward"]=[]
    info["history"]=[]
    info["cumulative reward"]=[]
    info["episode length"]=[]
    reward=np.zeros((1000,1000))
    if model_type=="discrete":
        reward=model.rewards()
    i=0
    k=0
    for x, a, r, y,done, truncated in history:
        if i==0:
            info["average reward"].append(r)
            info["cumulative reward"].append(r)
            if model_type=="discrete":
                info["average expected reward"].append(reward[x,a])
        else:
            info["cumulative reward"].append(r+info["cumulative reward"][-1])

        i+=1
        k+=1
        
        if done or truncated:
            info["episode length"].append(k)
            k=0
        
       
        info["average reward"   ].append((r + info["average reward"   ][-1]*(i-1))/i)
        if model_type=="discrete":
            info["average expected reward"   ].append((reward[x,a] + info["average expected reward"][-1]*(i-1))/i)

        info["history"].append([x,a,r,y])
    return info

################################################################################

class Agent:

    def __init__(self, model):
        """ The model is given as a parameter but stays unknown to the agent.
        It is mostly for convenience - e.g. to ease the writing of the code
        when rewards are known/unknown. 
        """

        # Initializing shape
        self.n_states=model.observation_space.n
        self.n_actions=model.action_space.n
        self.Z = set()
        for x in range(self.n_states):
            for a in range(self.n_actions):
                self.Z.add((x,a))
        self.S = list({s for s, _ in self.Z})
        self.A = [[] for _ in self.S]
        for s, a in self.Z:
            self.A[s].append(a)

        

    def reset(self, model):
        pass

    def name(self):
        pass

    def observe(self, x, a, r, y):
        pass

    def play(self, x):
        """ Pick an action """
        pass

class Random_Agent():
    def __init__(self,model,model_type):
        self.n_actions=model.action_space.n
        self.set_name("Random")
    def reset(self,model):
        pass
    def name(self):
        return self.name_str

    def set_name(self, name):
        self.name_str = name
    def observe(self,x,a,r,y,done,truncated):
        pass
    def act(self,x):
        return np.random.randint(self.n_actions)