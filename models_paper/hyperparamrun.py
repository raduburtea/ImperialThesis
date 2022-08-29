import sys
sys.path.append("./or-gym")
sys.path.append(".")
sys.path.append("./OMLT/src")
from tqdm import tqdm
import time
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from copy import deepcopy
import pyomo.environ as pyo
from omlt import OmltBlock, OffsetScaling
from omlt.io.keras import load_keras_sequential
from omlt.neuralnet import ReluBigMFormulation, FullSpaceNNFormulation, FullSpaceSmoothNNFormulation
import torch.onnx

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# from gym.wrappers import Monitor
# from gym.wrappers.monitoring.video_recorder import VideoRecorder

from actor import Actor
from critic import Critic
from replay import Transition, ReplayMemory
# from continuouscart import make_continuous_env
from noise import OrnsteinUhlenbeckActionNoise
from omlt_utils import optimise_actor_with_pyomo, make_deterministic
from omltddpg import one_session

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from or_gym.envs.supply_chain.inventory_management import InvManagementMasterEnv

import neptune.new as neptune

from omlt.io import write_onnx_model_with_bounds, load_onnx_neural_network_with_bounds
import tempfile


params = {
        "network_size_critic": (64,32),
        "network_size_actor": (32,32),
        "activation": "ReLU",
        "max_path_length": 30,
        "n_itr": 150,
        "gae_lambda": 0.95,
        "discount": 0.995,
        "BATCH_NORMAL": 512,
        "BATCH_OMLT": 8,
        "actor_lr": 0.001,
        "LR_OMLT": 0.01,
        "MLP_HIDDEN":'RELU',
        'activation_critic': 'sigmoid',
        'lr_critic': 0.001,
        "ALGO": 'OMLT',
        "Constraints": 'imposed',
        'solver': 'ipopt',
        'inventory': [100, 100, 200],
        'capacity': [100, 90, 80],
        'seed': 12
    }

# make_deterministic(params['seed'])
# run = None

NUM_EPISODES = 150
BATCH_SIZE = 64
GAMMA = 0.999
NOISE_START = 0.01
NOISE_END = 0
FINAL_NOISE_DECAY = 5   
NOISE_DECAY = NUM_EPISODES/FINAL_NOISE_DECAY     #number of episodes for epsilon to decay by 1/e


seeds = [60]
lr_actor_list = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
lr_critic_list = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
taus = [0.35, 0.1]
i = 0
for lr_actor in lr_actor_list:
    for lr_critic in lr_critic_list:
        for tau in taus:
            print('RUN ', i, lr_actor, lr_critic, tau)
            i+=1
            reward_list = []
            penalty_list = []
            
            run = neptune.init(
project="radu.burtea/DDPG-Hyperparamrun",
api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxYTEyNzhlOC1hZDVlLTQzZWQtOTYxZC1iMTNkNTgyMTg0Y2UifQ==",
)  # your credentials

            run['params'].log(params)
            
            for seed in seeds:
                params = {
                            "network_size_critic": (64,32),
                            "network_size_actor": (32,32),
                            "activation": "ReLU",
                            "max_path_length": 30,
                            "n_itr": 150,
                            "gae_lambda": 0.95,
                            "discount": 0.995,
                            "BATCH_NORMAL": 512,
                            "BATCH_OMLT": 8,
                            "actor_lr": lr_actor,
                            "LR_OMLT": 0.01,
                            "MLP_HIDDEN":'RELU',
                            'activation_critic': 'sigmoid',
                            'lr_critic': lr_critic,
                            "ALGO": 'DDPG',
                            "Constraints": 'imposed',
                            'solver': 'ipopt',
                            'inventory': [100, 100, 200],
                            'capacity': [100, 90, 80],
                            'seed': 60,
                            'tau': tau
                        }
                print(params['seed'])

                # torch.set_seed(params['seed'])
                # torch.backends.cudnn.deterministic = True
                random.seed(params['seed'])
                np.random.seed(params['seed'])
                torch.manual_seed(params['seed'])
                torch.cuda.manual_seed_all(params['seed'])
                # run = None
                env = InvManagementMasterEnv(None, max_path_length = params['max_path_length'], inventory = params['inventory'], capacity = params['capacity'])
                env.reset()
                state_dim = len(env.state)    #x, x_dot, theta, theta_dot
                action_dim = 3
                TAU = params['tau']  #target update
                hidden_layer_actor = 32
                hidden_layer_critic = 64
                actor_layers = (state_dim, hidden_layer_actor, 32, action_dim)
                critic_layers = (state_dim + action_dim, hidden_layer_critic, 32, 1)

                actor = Actor(actor_layers, lr = params['actor_lr'], max_force=1, device=device)
                #5e-3 before
                critic = Critic(critic_layers, lr = params['lr_critic'], device=device)

                episode_cutoff = 40000
                memory = ReplayMemory(60000)

                one_session(run, params, reward_list, penalty_list, episode_cutoff, NUM_EPISODES = NUM_EPISODES)

            
            # print(reward_list)
            
#             reward_list = np.array(reward_list)
#             rewards_means = np.mean(reward_list, axis = 0)
#             # print(rewards_means)
#             # print(np.mean(reward_list, axis = 0))
#             rewards_std = np.std(reward_list, axis = 0)
            
#             print(reward_list)
#             print(rewards_std)
            
#             penalties_list = np.array(penalty_list)
#             # print('Len of penalty list ', len(penalties_list))
#             penalties_means = np.mean(penalties_list, axis = 0)
#             penalties_std = np.std(penalties_list, axis = 0)
            # for i in range(len(penalties_list[0])):
            #     run['mean return'].log(rewards_means[i])
            #     run['std return'].log(rewards_std[i])
            #     run['mean penalty'].log(penalties_means[i])
            #     run['std penalty'].log(penalties_std[i])
            # print("lengths")
            # print('rewards.  ', len(penalties_std), len(rewards_std), len(rewards_means))
            
                