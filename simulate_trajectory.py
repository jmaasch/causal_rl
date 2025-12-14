from environment.mdp import MDP
from environment.generate_mdp import Generate_MDP
from policy.randompolicy import Random
import os
import pickle
from numpy.random import SeedSequence,default_rng
def main(n_statex,n_statey,n_action,n_experiments,n_replications_per_experiment,n_horizon,entropy):
    path='./simulate_trajectory'
    # ensure a path for the results
    path = os.path.abspath(path)
    os.makedirs(path, exist_ok=True)

    # construct the filename tag
    tag = "__".join(
        [
            
            f"X{n_statex}",
            f"Y{n_statey}",
            f"A{n_action}",
            f"E{n_experiments}",
            f"Re{n_replications_per_experiment}",
            f"H{n_horizon}",
        ]
    )
    data_pkl = os.path.join(path, f"data__{tag}.pkl")
     # create the main seed sequence
    main = SeedSequence(entropy)
    sq=main.spawn(n_experiments)
    results={}
    for e in range(n_experiments):
        results[f'experiment{e}']=[]
        sq_x_init,sq_y_init,sq_ker_x,sq_ker_y,sq_rew=sq[e].spawn(5)
        s_init=[]
        s_init.append(default_rng(sq_x_init).integers(n_statex))
        s_init.append(default_rng(sq_x_init).integers(n_statex))
        Generate=Generate_MDP(n_statex,n_statey,n_action)
        kernelx=Generate.generate_x(random=sq_ker_x)
        kernely=Generate.generate_y(random=sq_ker_y)
        reward=Generate.generate_reward(random=sq_rew)
        model=MDP(kernelx,kernely,reward)
        for i in range(n_replications_per_experiment):
            pi=Random(model)
            results[f'experiment{e}'].append(model.sample_path(s_init,pi,n_horizon))
    with open(data_pkl, "wb") as pkl:
            pickle.dump(results, pkl)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_statex",type=int)
    parser.add_argument("--n_statey",type=int)
    parser.add_argument("--n_action",type=int) 
    parser.add_argument("--n_experiments",type=int)
    parser.add_argument("--n_replications_per_experiment",type=int)      
    parser.add_argument("--n_horizon",type=int)      
    parser.add_argument("--entropy",type=int)   
    args = parser.parse_args()
    results = main(**vars(args))         


    
    