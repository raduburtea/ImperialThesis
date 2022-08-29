#This code is a modified version of the code used in Tutorial 6 of the 70028 - Reinforcement Learning  course 
#from Imperial College London

import torch
import torch.nn as nn
import torch.optim as optim

from copy import deepcopy

from mlp import MLP
import numpy as np

class Critic:
    def __init__(self, layers, lr=1e-3, device='cpu'):

        self.model = MLP(layers, output_activation = None, hidden_activation = nn.Sigmoid()).to(device)
        self.optim = optim.Adam(self.model.parameters(), lr=lr)

        self.target = deepcopy(self.model).to(device)
        self.target.eval()

        self.device = device

    def optimise_step(self, actor, transitions, non_final_next_states, non_final_mask, gradients, GAMMA, BATCH_SIZE):
        state, action, reward = transitions

        with torch.no_grad():
            non_final_next_state_action = torch.cat((non_final_next_states.to(self.device), actor.target(non_final_next_states.to(self.device))), dim=-1).detach().to(self.device)

            predicted_q_next = torch.zeros(len(reward)).to(self.device)

            predicted_q_next[non_final_mask] = self.target(non_final_next_state_action).squeeze()

        y = reward + GAMMA * predicted_q_next

        current_state_action = torch.cat((state.to(self.device), action.reshape(BATCH_SIZE,3).to(self.device)), dim=-1)
        q_current = self.model(current_state_action.to(self.device)).squeeze()

        loss = ((y - q_current)**2).mean()

        #optimisation step
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.3)
        grads = np.squeeze(self.model.layers[1].weight.grad.detach().cpu().numpy().reshape(-1,1))
        gradients.append(grads)
        self.optim.step()

    def update_target(self, tau):
        for target_param, local_param in zip(self.target.parameters(), self.model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
