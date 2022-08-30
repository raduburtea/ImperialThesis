import sys
sys.path.append("./or-gym")
sys.path.append(".")

import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from copy import deepcopy

import torch
from or_gym.envs.supply_chain.inventory_management import InvManagementMasterEnv
import neptune.new as neptune

from critic import Critic
from models_paper.omlt_utils import optimise_actor_with_pyomo, make_deterministic
import pickle 


#SET NAME OF ALGORITHM HERE
algo_name = 'SAFE-30'

#Path to the files constaining the saved models
#Sample saved models
critic_file_path = 'models_paper/models-OMLT/criticseed12p30normalnowarmup-qval.pt'
actor_file_path = 'models_paper/models-OMLT/actorseed12p30normalnowarmup-qval.pt'

if algo_name[:4] == 'RCPO':
    from rcpo import actor
else:
    from actor import Actor


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

params = {
    'algo': algo_name,
    'path_len': 30,
    'seed': 28,
    'inventory': [100, 100, 200],
    'capacity': [100, 90, 80],
    'episodes': 50,
    'use_neptune': False
    }
    
if params['use_neptune']:
    #Please include your neptune.ai credentials
    run = neptune.init(
        project="sample_name",
        api_token="sample_token",
    )  # your credentials

    run['params'].log(params)
else:
    run = None

#Flag is needed to detach the tensor and convert it to numpy is algorithm is not CPO or TRPO
env = InvManagementMasterEnv(None, max_path_length = params['path_len'], inventory = params['inventory'], capacity = params['capacity'],  \
                            flag = params['algo'][:4] not in ['CPO-', 'TRPO'])


env.reset()
state_dim = len(env.state)
action_dim = 3

make_deterministic(params['seed'])

#Initialiaze actor and critic
hidden_layer_actor = 32
hidden_layer_critic = 64
actor_layers = (state_dim, hidden_layer_actor, 32, action_dim)
critic_layers = (state_dim + action_dim, hidden_layer_critic, 32, 1)

#Different initialization for the actors based on the algorithm used 
if algo_name[:4] != 'RCPO':
    actor = Actor(actor_layers, lr = 0.001, max_force=1, device=device)

if params['algo'][:4] in ['CPO-', 'TRPO']:
    actor = pickle.load(open(actor_file_path, 'rb'))

if params['algo'][:4] not in ['CPO-', 'TRPO']:
    actor_dict = torch.load(actor_file_path)
    actor.model.load_state_dict(actor_dict)
    actor.model.eval()

    #Loading critic 
    critic = Critic(critic_layers, lr = 0.001, device=device)
    critic_dict = torch.load(critic_file_path)
    critic.model.load_state_dict(critic_dict)
    critic.model.eval()

episodes = params['episodes']

#Running the actual simulation
for episode in range(episodes):
    env.reset()
    state = torch.tensor(env.state).float().unsqueeze(0)
    rewards = []
    actions = []
    penalties = []
    status_ep = {'Node1/Replenishment order': [], 'Node2/Replenishment order': [], 'Node3/Replenishment order': [],
            'Node1/Inventory constraint': [], 'Node2/Inventory constraint': [], 'Node1/Capacity constraint': [], 'Node2/Capacity constraint': [],
            'Node3/Capacity constraint': [], 'Node1/Inventory constraint next': [], 'Node2/Inventory constraint next': []}
    for t in count():

        if algo_name[:4] == 'RCPO':
            actor_action = actor(state.to(device)).sample().to(device)[0]
        elif algo_name[:4] in ['DDPG', 'OMLT']:

            actor_action = actor.model(state.to(device)).flatten().detach()

        elif algo_name[:3] in ['CPO', 'TRP']:
            actor_action = actor.get_action(state)[0]

        #Implementation of the SAFE algorithm for Safe Exploitation
        elif algo_name[:4] == 'SAFE':
            if t > 1:
                #Workaround for the case when Pyomo fails with unexpected errors
                #The optimization problem will be re-run for 4 times and if it fails for all of the attepmts
                #the action from the actor will be taken
                for attempt_no in range(4):
                    try:
                        initial_guess = actor.model(state.to(device)).flatten().detach()
                        actor_action = optimise_actor_with_pyomo(actor, critic, state, initial_guess, device, node_status,  BATCH_SIZE = 1, mode='test')[0]
                        break
                    except:
                        if attempt_no < 3:
                             print("Retrying the optimization problem")
                        else:
                            actor_action = actor.model(state.to(device)).flatten().detach()
            else:
                actor_action = actor.model(state.to(device)).flatten().detach()

        _, reward, done, penalty, node_status, actual_action = env.step(actor_action)
        rewards.append(reward)
        penalties.append(penalty)
        reward = torch.tensor([reward], device=device)

        # Observe new state
        if not done:
            next_state = torch.tensor(env.state).float().unsqueeze(0)
        else:
            next_state = None
            
        if run:
            for status in node_status.keys():
                        run[status].log(node_status[status])

        # Move to the next state
        state = next_state

        #End episode
        if done:
            print('Reward for episode %s' % str(episode), np.sum(rewards))
            print('Penalty for episode ', np.sum(penalties))
            if run:
                run['Reward'].log(np.sum(rewards))
                run['Penalty'].log(np.sum(penalties))
            break
