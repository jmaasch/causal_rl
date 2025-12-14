# causal_rl
Preliminary experiments for causal RL.

Run the following for results. State space is 10*10, action is binary. There are total 10 different instances and each instance is replicated for 100 times (change n_replications_per_experiment for larger). The time horizon is 1000. In the pkl file, results is a dictionary. results['experimentx'] is the list of all 100 different trajectories for instance x.

 python simulate_trajectory.py --n_statex 10 --n_statey 10 --n_action 2 --n_experiments 10 --n_replications_per_experiment 100 --n_horizon 1000 --entropy 243799254704924441050048792905230269161

