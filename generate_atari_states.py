import gym
from atariari.benchmark.wrapper import AtariARIWrapper
import pickle
import os
from numpy.random import SeedSequence,default_rng
import tqdm
atari_dict={ "venture": dict(sprite0_y=20,
                    sprite1_y=21,
                    sprite2_y=22,
                    sprite3_y=23,
                    sprite4_y=24,
                    sprite5_y=25,
                    sprite0_x=79,
                    sprite1_x=80,
                    sprite2_x=81,
                    sprite3_x=82,
                    sprite4_x=83,
                    sprite5_x=84,
                    player_x=85,
                    player_y=26,
                    current_room=90,  # The number of the room the player is currently in 0 to 9_
                    num_lives=70,
                    score_1_2=71,
                    score_3_4=72)}
results={}
path='./simulate_trajectory'
# ensure a path for the results
path = os.path.abspath(path)
os.makedirs(path, exist_ok=True)
data_pkl = os.path.join(path, f"data__Venture__10000.pkl")
ss = SeedSequence(12345)

# generate N independent seeds
child_seeds = ss.spawn(10000)

# convert each to an integer seed
seeds = [s.generate_state(1)[0] for s in child_seeds]
for e in range(10000):  
    env = gym.make('VentureNoFrameskip-v4')
    obs=env.reset()
    env.seed(seeds[e])
    results[e]=[]
    for T in tqdm.tqdm(range(100), desc=str(e)):
        action=env.action_space.sample()
        obs,reward,terminated,truncated,info=env.step(action)
        ram = env.unwrapped.ale.getRAM()
        info["labels"]={k: ram[ind] for k, ind in atari_dict['Venture'.lower()].items()}
        results[e].append((action,info))
        if terminated or truncated:
            break
print(results[0])
with open(data_pkl, "wb") as pkl:
            pickle.dump(results, pkl)


