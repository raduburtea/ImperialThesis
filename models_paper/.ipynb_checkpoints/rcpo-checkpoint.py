# Code adapted from https://github.com/hermesdt/reinforcement-learning/blob/master/a2c/pendulum_a2c_online.ipynb
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import sys
sys.path.append("./or-gym")
sys.path.append(".")
from or_gym.envs.supply_chain.inventory_management import InvManagementMasterEnv
import neptune.new as neptune
from omlt_utils import make_deterministic

def mish(input):
    """Mish activation function"""
    return input * torch.tanh(F.softplus(input))

class Mish(nn.Module):
    """Class that implements the Mish activation function"""
    def __init__(self): super().__init__()
    def forward(self, input): return mish(input)

params = {
        "lagrangian_lr": 5e-5,
        "n_itr": 4000,
        "max_path_len":15,
        "discount": 0.9,
        'seed': 629, 
        "lr_actor": 0.0001,
        "lr_critic": 0.001,
        'inventory': [80, 80, 120],
        'capacity': [80, 75, 65],
        'max_grad_norm': 0.2,
        'entropy_beta': 0,
        'use_neptune': False
    }


make_deterministic(params['seed'])

#If the GPU is outdated but CUDA still shows as switch to CPU by uncommenting the following line
device = torch.device("cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# helper function to convert numpy arrays to tensors
def t(x, device):
    x = np.array(x) if not isinstance(x, np.ndarray) else x
    return torch.from_numpy(x).float().to(device)


#Actor module 
class Actor(nn.Module):
    def __init__(self, state_dim, n_actions, activation=nn.Tanh):
        super().__init__()
        self.n_actions = n_actions
        self.model = nn.Sequential(
            nn.Linear(state_dim, 128),
            activation(),
            nn.Linear(128, 64),
            activation(),
            nn.Linear(64, n_actions)
        )
        
        logstds_param = nn.Parameter(torch.full((n_actions,), 0.1))
        self.register_parameter("logstds", logstds_param)
    
    #Forward method of Gaussian MLP
    def forward(self, X):
        means = self.model(X)
        stds = torch.clamp(self.logstds.exp(), 1e-3, 50)
        
        return torch.distributions.Normal(means, stds)

## Critic module
class Critic(nn.Module):
    def __init__(self, state_dim, activation=nn.Tanh):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            activation(),
            nn.Linear(256, 64),
            activation(),
            nn.Linear(64, 1),
        )
    
    def forward(self, X):
        return self.model(X)


def discounted_penalized_rewards(rewards, penalties, lagrangian, dones, gamma):
    """Computed the penalized discounted rewards for each step in an episode"""
    ret = 0
    discounted = []
    for reward, penalty, done in zip(rewards[::-1], penalties[::-1], dones[::-1]):
        ret = reward + ret * gamma * (1-done) - lagrangian * penalty
        discounted.append(ret)
    
    return discounted[::-1]

def process_memory(memory, lagrangian, device, gamma=0.99, discount_rewards=True):
    """Process an entire episode by adding the penalized discounted rewards and transforming each
    variable in memory into a torch tensor"""
    actions = []
    states = []
    next_states = []
    rewards = []
    penalties = []
    dones = []

    for action, reward, state, next_state, penalty, done in memory:
        actions.append(action)
        rewards.append(reward)
        states.append(state)
        next_states.append(next_state)
        penalties.append(penalty)
        dones.append(done)
    
    #Obtain disocunted reward for each step in the path 
    if discount_rewards:
        if False and dones[-1] == 0:
            rewards = discounted_penalized_rewards(rewards + [last_value], penalties, lagrangian, dones + [0], gamma)[:-1]
        else:
            rewards = discounted_penalized_rewards(rewards, penalties, lagrangian, dones, gamma)

    actions = torch.cat(actions).reshape(params['max_path_len'],3).to(device)
    states = torch.tensor(states).to(device)
    next_states = torch.tensor(next_states).to(device)
    rewards = torch.tensor(rewards).view(-1, 1).to(device)
    penalties = torch.tensor(penalties).view(-1, 1).to(device)
    dones = torch.tensor(dones).view(-1, 1).to(device)
    return actions, rewards, states, next_states, penalties, dones

def clip_grad_norm_(module, max_grad_norm):
    """"Clip the gradients to prevent gradient explosion"""
    nn.utils.clip_grad_norm_([p for g in module.param_groups for p in g["params"]], max_grad_norm)

class A2CLearner():
    """Class that performs the optimization for the actor and critic of the A2C architecture"""
    def __init__(self, actor, critic, lr_lagrangian=1e-4, gamma=0.9, entropy_beta=0,
                 actor_lr=1e-4, critic_lr=1e-3, max_grad_norm=0.2):
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.actor = actor
        self.critic = critic
        self.alpha = 0.1
        self.entropy_beta = entropy_beta
        self.actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(critic.parameters(), lr=critic_lr)
        self.lagrangian = 0
        self.lr_lagrangian = lr_lagrangian
    
    def learn(self, memory, device, discount_rewards=True):
        """Optimize actor and critic"""
        actions, rewards, states, next_states, penalties, dones = process_memory(memory, self.lagrangian, device, self.gamma, discount_rewards=True)

        #Compute td target
        td_target = rewards

        value = critic(states.float().to(device))
        advantage = td_target - value

        # actor
        norm_dists = self.actor(states.float())
        # Compute the log of the probability density function and entropy of the distribution 
        logs_probs = norm_dists.log_prob(actions)
        entropy = norm_dists.entropy().mean()
        
        #Compute the loss for the actor by taking the negative value of the product between the advantages 
        #and the log probability. From this the product of the entropy and the hyperparameter beta is deducted 
        #in order to further induce uncertainty if the distribution is too homogenous
        actor_loss = (-logs_probs*advantage.detach().to(device)).mean() - entropy*self.entropy_beta
        self.actor_optim.zero_grad()
        actor_loss.backward()
        
        #Clip gradients of the actor
        clip_grad_norm_(self.actor_optim, self.max_grad_norm)
        if neptune_run:
            neptune_run["gradients/actor"].log(
                                 torch.cat([p.grad.view(-1) for p in self.actor.parameters()]) )
            neptune_run["parameters/actor"].log(
                                 torch.cat([p.data.view(-1) for p in self.actor.parameters()]) )
        #Perform optimization step for the actor
        self.actor_optim.step()
        
        #lagrangian
        self.lagrangian = max(0, self.lagrangian + self.lr_lagrangian*(penalties.mean() - self.alpha))
        
        # Compute loss for the critic and run optimization step
        critic_loss = F.mse_loss(td_target.double().to(device), value.double().to(device))
        self.critic_optim.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(self.critic_optim, self.max_grad_norm)
        if neptune_run:
            neptune_run["gradients/critic"].log(
                                 torch.cat([p.grad.view(-1) for p in self.critic.parameters()]) )
            neptune_run["parameters/critic"].log(
                                 torch.cat([p.data.view(-1) for p in self.critic.parameters()]) )
        self.critic_optim.step()
        
        if neptune_run:
            neptune_run["penalized reward"].log(rewards.sum())
            neptune_run["losses/log_probs"].log(-logs_probs.mean())
            neptune_run["losses/entropy"].log(entropy) 
            neptune_run["losses/entropy_beta"].log(self.entropy_beta ) 
            neptune_run["losses/actor"].log(actor_loss )
            neptune_run["losses/advantage"].log(advantage.mean() )
            neptune_run["losses/critic"].log(critic_loss )
            neptune_run['Lagrangian'].log(self.lagrangian)

class Runner():
    """Class that runs the simulation"""
    def __init__(self, env):
        self.env = env
        self.state = None
        self.done = True
        self.steps = 0
        self.episode_reward = 0
        self.episode_rewards = []
    
    def reset(self):
        """Resets the environment to initial values"""
        self.episode_reward = 0
        self.done = False
        self.state = self.env.reset()
    
    def run(self, index, max_steps, rewards, device, memory=None):
        """Runs the simulation"""
        if not memory: memory = []
        status_ep = {'Node1/Replenishment order': [], 'Node2/Replenishment order': [], 'Node3/Replenishment order': [],
            'Node1/Inventory constraint': [], 'Node2/Inventory constraint': [], 'Node1/Capacity constraint': [], 'Node2/Capacity constraint': [],
            'Node3/Capacity constraint': [],  'Node1/Backlogs': [], 'Node2/Backlogs': [], 'Node3/Backlogs': [], 'Node1/LostSales': [], 'Node2/LostSales': [], 'Node3/LostSales': [], 'Node1/Inventory constraint next': [], 'Node2/Inventory constraint next': [], 'Penalty':[]}

        for i in range(max_steps):

            if self.done:
                self.reset()
            
            #Obtain distribution of actor network for a given state
            dists = actor(t(self.state, device))
            #Sample action from the distribution
            actions = dists.sample().to(device)

            penalties = []
            #Take step
            next_state, reward, self.done, penalty, node_status, _ = self.env.step(actions)

            penalties.append(penalty)
            memory.append((actions, reward, self.state, next_state, penalty, self.done))
            
            for status in node_status.keys():
                    status_ep[status].append(node_status[status])

            self.state = next_state
            self.steps += 1
            self.episode_reward += reward
            
            if self.done:
                self.episode_rewards.append(self.episode_reward)
                if index % 10 == 0:
                    print("episode:", index, ", episode reward:", self.episode_reward)
                    print('Penalty ', np.mean(penalties))
                rewards.append(self.episode_reward)
                if neptune_run:
                     neptune_run['reward'].log(self.episode_reward)                     
                     neptune_run['Penalty'].log(np.mean(penalties))
                     neptune_run['Node1/Replenishment order'].log(np.mean(status_ep['Node1/Replenishment order']))
                     neptune_run['Node2/Replenishment order'].log(np.mean(status_ep['Node2/Replenishment order']))
                     neptune_run['Node3/Replenishment order node 3'].log(np.mean(status_ep['Node3/Replenishment order']))
                     neptune_run['Node1/Inventory constraint'].log(np.mean(status_ep['Node1/Inventory constraint']))
                     neptune_run['Node2/Inventory constraint'].log(np.mean(status_ep['Node2/Inventory constraint']))
                     neptune_run['Node1/Capacity constraint'].log(np.mean(status_ep['Node1/Capacity constraint']))
                     neptune_run['Node2/Capacity constraint'].log(np.mean(status_ep['Node2/Capacity constraint']))
                     neptune_run['Node3/Capacity constraint'].log(np.mean(status_ep['Node3/Capacity constraint']))
        
        return memory


env = InvManagementMasterEnv(None, max_path_length = params['max_path_len'], inventory = params['inventory'], capacity = params['capacity'])
    
state_dim = env.observation_space.shape[0]
n_actions = 3
steps_on_memory = params['max_path_len']


rewards = []

actor = Actor(state_dim, n_actions, activation=Mish).to(device)
critic = Critic(state_dim, activation=Mish).to(device)

learner = A2CLearner(actor, critic, lr_lagrangian=params['lagrangian_lr'], gamma=params['discount'], entropy_beta=params['entropy_beta'], \
                        actor_lr=params['lr_actor'], critic_lr=params['lr_critic'], max_grad_norm=params['max_grad_norm'])
runner = Runner(env)

if __name__ == '__main__':

    if params['use_neptune']:
        #Please include your neptune.ai credentials
        neptune_run = neptune.init(
            project="sample_name",
            api_token="sample_token",
        )  # your credentials

        neptune_run['params'].log(params)
    else:
        neptune_run = None 



    #Run training for the desired number of episodes
    for i in range(params['n_itr']):
        memory = runner.run(i, params['max_path_len'], rewards, device)
        learner.learn(memory, device)