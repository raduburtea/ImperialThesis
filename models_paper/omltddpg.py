#This code is a modified version of the code used in Tutorial 6 of the 70028 - Reinforcement Learning  course 
#from Imperial College London

import sys
sys.path.append("./or-gym")
sys.path.append(".")
sys.path.append("./OMLT/src")
from tqdm import tqdm
import numpy as np
from itertools import count
import torch.onnx

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from actor import Actor
from critic import Critic
from replay import Transition, ReplayMemory
from omlt_utils import optimise_actor_with_pyomo, make_deterministic
from or_gym.envs.supply_chain.inventory_management import InvManagementMasterEnv
import neptune


#If you want to run the OMLT-DDPG variant without warm-up pass 0 to the warm_up_cutoff variable
# Otherwise pass any other integer that would determine the number of episodes used for warm-up, 30 have been used initially
#In order to run the DDPG algorithm pass the name DDPG to the ALGO variable
params = {
        'save_name': 'seed28p15normalnowarmup-qval',
        "max_path_length": 15,
        "use_neptune": False,
        'activation_critic': 'relu',
        "ALGO": 'OMLT',
        'inventory': [100, 100, 200],
        'capacity': [100, 90, 80],
        "warm_up_cutoff":0,
        "n_itr": 150,
        "discount": 0.995,
        "BATCH_NORMAL": 512,
        "BATCH_OMLT": 8,
        "actor_lr": 1e-5,
        "LR_OMLT": 0.005,
        'lr_critic': 0.001,
        'solver': 'ipopt',
        'seed': 28,
        'tau': 0.35,
        'gamma':0.999
    }

make_deterministic(params['seed']) #set seed across all packages that use random generation

device = torch.device("cpu")
#If you want to used GPU, but only for DDPG, uncomment the following line
# device = torch.device("cuda")

NUM_EPISODES = params['n_itr']
BATCH_SIZE = params['BATCH_NORMAL'] #batch used for critic training 
GAMMA = params['gamma'] #discount factor
TAU = params['tau'] #target update

#Initialize the environment
env = InvManagementMasterEnv(None, max_path_length = params['max_path_length'], inventory = params['inventory'], capacity = params['capacity'])
env.reset()

# Get number of states and actions from gym action space
state_dim = len(env.state)  
action_dim = 3

#Initialize Actor and Critic Networks
hidden_layer_actor = 32
hidden_layer_critic =64
actor_layers = (state_dim, hidden_layer_actor, 32, action_dim)
critic_layers = (state_dim + action_dim, hidden_layer_critic, 32, 1)

actor = Actor(actor_layers, lr = params['actor_lr'], max_force=1, device=device)
critic = Critic(critic_layers, lr = params['lr_critic'], device=device)

warm_up_cutoff = params['warm_up_cutoff']
memory = ReplayMemory(60000)

run = None
def optimise_models(actor, critic, memory, episode, node_status, critic_params, gradients, run, warm_up_cutoff=warm_up_cutoff, GAMMA=GAMMA, BATCH_SIZE=BATCH_SIZE):
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)

    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    node_batch = batch.node_status

    critic.optimise_step(actor, (state_batch, action_batch, reward_batch),
                         non_final_next_states, non_final_mask, gradients, GAMMA=GAMMA, BATCH_SIZE=BATCH_SIZE)




    if params['ALGO'] == 'DDPG':
        actor.optimise_step(critic, state_batch, episode)

    else:
        if episode > warm_up_cutoff:
            #Exception in case the Pyomo solver throws out exception for unknown reasons - rare occurence
            try:
                act, loss_actor = optimise_actor_with_pyomo(actor, critic, state_batch, action_batch, device, \
                     node_batch, BATCH_SIZE = params['BATCH_OMLT'], solver_name = params['solver'])
                actor.optim = optim.Adam(actor.model.parameters(), lr = params["LR_OMLT"])
            except KeyboardInterrupt:
                raise Exception()
            except:
                print("Pyomo solver threw an unexpected error")




def start_simulation(run, params, returns_all_eps, penalties_all_eps, warm_up_cutoff=warm_up_cutoff, NUM_EPISODES = NUM_EPISODES):
    returns_eps = []
    penalties_eps = []
    critic_params = []
    gradients = []
    actions_full = []
    episode_durations = []

    for i_episode in tqdm(range(NUM_EPISODES)):

        if i_episode < warm_up_cutoff:
            BATCH_SIZE = params["BATCH_NORMAL"]
        else:
            BATCH_SIZE = params["BATCH_OMLT"]

        ep_actions = []
        ep_noise = []
        rewards_all_episodes = []

        if i_episode % 5 == 0:
            print("episode ", i_episode, "/", NUM_EPISODES)

        # Initialize the environment and state
        env.reset()
        state = torch.tensor(env.state).float().unsqueeze(0)

        rewards = []
        actions = []
        penalties = []

        status_ep = {'Node1/Replenishment order': [], 'Node2/Replenishment order': [], 'Node3/Replenishment order': [],
                'Node1/Inventory constraint': [], 'Node2/Inventory constraint': [], 'Node1/Capacity constraint': [], 'Node2/Capacity constraint': [],
                'Node3/Capacity constraint': [],  'Node1/Backlogs': [], 'Node2/Backlogs': [], 'Node3/Backlogs': [], 'Node1/LostSales': [], 'Node2/LostSales': [],
                 'Node3/LostSales': [], 'Node1/Inventory constraint next': [], 'Node2/Inventory constraint next': [], 'Penalty':[]}
        for t in count():
            actor_action = actor.model(state.to(device)).flatten().detach() 
            if params['use_neptune']:
                run['actor_action'].log(float(np.mean(actor_action.cpu().numpy())))
                        
            _, reward, done, penalty, node_status, actual_action = env.step(actor_action)
            penalties.append(penalty)
            rewards.append(reward)
            actions.append(actor_action.cpu().numpy())
            reward = torch.tensor([reward], device=device)

            # Observe new state
            if not done:
                next_state = torch.tensor(env.state).float().unsqueeze(0)
            else:
                next_state = None

            for status in node_status.keys():
                        status_ep[status].append(node_status[status])
            
            # Store transition in memory
            memory.push(state, actor_action, next_state, reward, node_status)

            # Move to the next state
            state = next_state

            # Optimise actor and critic
            optimise_models(actor, critic, memory, i_episode, node_status, critic_params, gradients, run, warm_up_cutoff = warm_up_cutoff, BATCH_SIZE=BATCH_SIZE)

            #Update target networks
            actor.update_target(TAU)
            critic.update_target(TAU)
            rewards_all_episodes.append(np.sum(rewards))
            #End episode
            if done:
                critic_params.append(np.squeeze(critic.model.layers[0].weight.detach().cpu().numpy().reshape(-1,1)))
                if run:
                    run["Return"].log(np.sum(rewards))
                    run['Penalties'].log(np.mean(penalties))
                    run['Node1/Replenishment order'].log(np.mean(status_ep['Node1/Replenishment order']))
                    run['Node2/Replenishment order'].log(np.mean(status_ep['Node2/Replenishment order']))
                    run['Node3/Replenishment order node 3'].log(np.mean(status_ep['Node3/Replenishment order']))
                    run['Node1/Inventory constraint'].log(np.mean(status_ep['Node1/Inventory constraint']))
                    run['Node2/Inventory constraint'].log(np.mean(status_ep['Node2/Inventory constraint']))
                    run['Node1/Capacity constraint'].log(np.mean(status_ep['Node1/Capacity constraint']))
                    run['Node2/Capacity constraint'].log(np.mean(status_ep['Node2/Capacity constraint']))
                    run['Node3/Capacity constraint'].log(np.mean(status_ep['Node3/Capacity constraint']))
                    
                    run['Node1/Backlogs'].log(np.mean(status_ep['Node1/Backlogs']))
                    run['Node2/Backlogs'].log(np.mean(status_ep['Node2/Backlogs']))
                    run['Node1/LostSales'].log(np.mean(status_ep['Node1/LostSales']))
                    run['Node2/LostSales'].log(np.mean(status_ep['Node2/LostSales']))

                episode_durations.append(t + 1)
                rewards_all_episodes.append(np.sum(rewards))

                if i_episode > warm_up_cutoff:
                    actions_full.append(np.mean(np.array(actions), axis =1))
                    
                if i_episode % 15 == 0:
                    print("duration  ", episode_durations[-1])
                    print("rewards ", np.sum(rewards))
                    print("Penalties ", np.mean(penalties))
                
                returns_eps.append(np.sum(rewards))
                penalties_eps.append(np.mean(penalties))
                break
    returns_all_eps.append(np.array(returns_eps))
    penalties_all_eps.append(np.array(penalties_eps))



if __name__ == '__main__':

    
    if params['use_neptune']:
        #Please include your neptune.ai credentials
        run = neptune.init(
            project="sample_name",
            api_token="sample_token",
        )  # your credentials

        run['params'].log(params)
        
        start_simulation(run, params, [], [])
    else:
        #If neptune.ai is not used pass None for run, otherwise pass run
        start_simulation(None, params, [], [])

