from __future__ import division
from policies.agent import iterate_algorithm, parse_history,Random_Agent
from policies.memory import ReplayMemory
from policies.agent_dqn import Agent
import os
import pickle
from numpy.random import SeedSequence,default_rng
from functools import partial
from itertools import product
import numpy as np
import random
from datetime import datetime
import tqdm
from matplotlib import pyplot as plt
import gymnasium as gym
import time
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import ale_py
import torch

import argparse
def time_str(sec):
    s = int(sec)
    m = s // 60
    h = m // 60
    m = m % 60
    s = s % 60
    if h: 
        return f"{h}h{m:2d}m{s:2d}s"
    if m:
        return f"{m}m{s:2d}s"
    return f"{s}s"
def main(args):
    path='./results_random'
    path = os.path.abspath(path)
    os.makedirs(path, exist_ok=True)
    all_envs = gym.registry.keys()

# Filter Atari ALE environments
    atari_envs = [
    env_id for env_id in all_envs
    if env_id.startswith("ALE/") and env_id.endswith("-v5")
]
    subset = random.sample(atari_envs, 20)
    state_stack=args.history_length
    for game in subset:
        game_name=game.split("/")[1].split("-")[0]
        data_pkl = os.path.join(path, f"data__rainbow_{game_name}.pkl")
        ss = SeedSequence(243799254704924441050048792905230269161)
        sq=ss.spawn(args.n_replications)
        torch.manual_seed(np.random.randint(1, 10000))
        if torch.cuda.is_available() and not args.disable_cuda:
            args.device = torch.device('cuda')
            torch.cuda.manual_seed(np.random.randint(1, 10000))
            torch.backends.cudnn.enabled = args.enable_cudnn
        else:
            args.device = torch.device('cpu')
        t0=time.time()
        
        gym.register_envs(ale_py)
        for e in range(args.n_replications):
            t_spend = time.time() - t0
            t_rem = (args.n_replications - e) * t_spend/max(e, 1)
            print(f"Run {e+1} ... (spend {time_str(t_spend)}, remains {time_str(t_rem)})")
            
            if e==0:
                    results={f"rainbow(stack={n})": {"average_output": np.zeros((args.T_max,args.n_replications))} for n in state_stack}
            for n in state_stack:
                args.history_length=n
                model=gym.make(game,frameskip=1)
                model = AtariPreprocessing(model, grayscale_obs=True, scale_obs=False, frame_skip=4)
                model = FrameStackObservation(model, stack_size=n)  # returns (84,84,4) or stacked object depending on version


                dqn=Agent(args,env=model)
                action_space=model.action_space.n
                mem = ReplayMemory(args, args.memory_capacity)
                priority_weight_increase = (1 - args.priority_weight) / (args.T_max - args.learn_start)
                
                dqn.train()
                done=True
                truncated=True
                history=[]
                for T in tqdm.tqdm(range(args.T_max), desc="rainbow"):
                    if done or truncated:
                        state,_=model.reset(seed=int(sq[e].generate_state(1)[0]))
                        state = torch.as_tensor(state, device=torch.linspace(args.V_min, args.V_max, args.atoms).to(device=args.device).device, dtype=torch.float32)
                    if T % args.replay_frequency == 0:
                        dqn.reset_noise()  # Draw a new set of noisy weights
                    action = dqn.act(state)  # Choose an action greedily (with noisy weights)
                    state = torch.as_tensor(state, device=torch.linspace(args.V_min, args.V_max, args.atoms).to(device=args.device).device, dtype=torch.float32)
                    next_state, reward, done,truncated,_ = model.step(action)  # Step
                    next_state = torch.as_tensor(next_state, device=torch.linspace(args.V_min, args.V_max, args.atoms).to(device=args.device).device, dtype=torch.float32)
                    history.append((state,action,reward,next_state,done,truncated))

                    if args.reward_clip > 0:
                        reward = max(min(reward, args.reward_clip), -args.reward_clip)  # Clip rewards
                    mem.append(state, action, reward, done)  # Append transition to memory
                    if T >= args.learn_start:
                        mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)  # Anneal importance sampling weight β to 1

                        if T % args.replay_frequency == 0:
                            dqn.learn(mem)  # Train with n-step distributional double-Q learning
                        if T % args.target_update == 0:
                            dqn.update_target_net()
                    state=next_state
                info=parse_history(model,history,model_type="continuous")
                # print(len(info["cumulative reward"]),results["rainbow"]["average_output"][:,e].shape)
                results[f"rainbow(stack={n})"]["average_output"][:,e]=info["cumulative reward"]

            with open(data_pkl, "wb") as pkl:
                    pickle.dump(results, pkl)
            fig, ax = plt.subplots()
            legend=[]
            colors = plt.cm.tab10.colors
            for n,col in zip(state_stack,colors):
                std=np.std(results[f"rainbow(stack={n})"]["average_output"],axis=-1,ddof=1)
                Y=np.mean(results[f"rainbow(stack={n})"]["average_output"],axis=-1)
                line,= ax.plot(Y,color=col,label=f"rainbow(stack={n})")
                ax.fill_between(np.arange(args.T_max), Y-1.96*std/np.sqrt(args.n_replications), Y+1.96*std/np.sqrt(args.n_replications), color=col, alpha=0.2, linewidth=0)
                legend.append((line,f"rainbow(stack={n})"))
            fig.legend(*zip(*legend), loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=5,fontsize=12)

            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(f"test_{game_name}_dqn.pdf", bbox_inches="tight")
        # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rainbow')
    parser.add_argument('--id', type=str, default='default', help='Experiment ID')
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    # parser.add_argument('--game', type=str, default='space_invaders', choices=atari_py.list_games(), help='ATARI game')
    parser.add_argument('--T-max', type=int, default=int(1000), metavar='STEPS', help='Number of training steps (4x number of frames)')
    parser.add_argument('--max-episode-length', type=int, default=int(108e3), metavar='LENGTH', help='Max episode length in game frames (0 to disable)')
    parser.add_argument('--history-length', type=int, default=[2,4,8,16], metavar='T', help='Number of consecutive states processed')
    parser.add_argument('--architecture', type=str, default='canonical', choices=['canonical', 'data-efficient'], metavar='ARCH', help='Network architecture')
    parser.add_argument('--hidden-size', type=int, default=512, metavar='SIZE', help='Network hidden size')
    parser.add_argument('--noisy-std', type=float, default=0.1, metavar='σ', help='Initial standard deviation of noisy linear layers')
    parser.add_argument('--atoms', type=int, default=51, metavar='C', help='Discretised size of value distribution')
    parser.add_argument('--V-min', type=float, default=-10, metavar='V', help='Minimum of value distribution support')
    parser.add_argument('--V-max', type=float, default=10, metavar='V', help='Maximum of value distribution support')
    parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
    parser.add_argument('--memory-capacity', type=int, default=int(1e6), metavar='CAPACITY', help='Experience replay memory capacity')
    parser.add_argument('--replay-frequency', type=int, default=4, metavar='k', help='Frequency of sampling from memory')
    parser.add_argument('--priority-exponent', type=float, default=0.5, metavar='ω', help='Prioritised experience replay exponent (originally denoted α)')
    parser.add_argument('--priority-weight', type=float, default=0.4, metavar='β', help='Initial prioritised experience replay importance sampling weight')
    parser.add_argument('--multi-step', type=int, default=3, metavar='n', help='Number of steps for multi-step return')
    parser.add_argument('--discount', type=float, default=0.99, metavar='γ', help='Discount factor')
    parser.add_argument('--target-update', type=int, default=int(1), metavar='τ', help='Number of steps after which to update target network')
    parser.add_argument('--reward-clip', type=int, default=1, metavar='VALUE', help='Reward clipping (0 to disable)')
    parser.add_argument('--learning-rate', type=float, default=0.0000625, metavar='η', help='Learning rate')
    parser.add_argument('--adam-eps', type=float, default=1.5e-4, metavar='ε', help='Adam epsilon')
    parser.add_argument('--batch-size', type=int, default=8, metavar='SIZE', help='Batch size')
    parser.add_argument('--norm-clip', type=float, default=10, metavar='NORM', help='Max L2 norm for gradient clipping')
    parser.add_argument('--learn-start', type=int, default=int(64), metavar='STEPS', help='Number of steps before starting training')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
    parser.add_argument('--evaluation-interval', type=int, default=100000, metavar='STEPS', help='Number of training steps between evaluations')
    parser.add_argument('--evaluation-episodes', type=int, default=10, metavar='N', help='Number of evaluation episodes to average over')
    # TODO: Note that DeepMind's evaluation method is running the latest agent for 500K frames ever every 1M steps
    parser.add_argument('--evaluation-size', type=int, default=500, metavar='N', help='Number of transitions to use for validating Q')
    parser.add_argument('--render', action='store_true', help='Display screen (testing only)')
    parser.add_argument('--enable-cudnn', action='store_true', help='Enable cuDNN (faster but nondeterministic)')
    parser.add_argument('--checkpoint-interval', default=0, help='How often to checkpoint the model, defaults to 0 (never checkpoint)')
    parser.add_argument('--memory', help='Path to save/load the memory from')
    parser.add_argument('--disable-bzip-memory', action='store_true', help='Don\'t zip the memory file. Not recommended (zipping is a bit slower and much, much smaller)')
    parser.add_argument('--n_replications',type=int,default=50)

    # Setup
    args = parser.parse_args()
    main(args)





