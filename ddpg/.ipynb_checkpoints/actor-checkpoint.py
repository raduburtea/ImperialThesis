import torch
import torch.nn as nn
import torch.optim as optim

from copy import deepcopy

from mlp import MLP

class Actor:
    def __init__(self, layers, lr, max_force=1, device='cpu'):

        self.model = MLP(layers, output_activation='relu', max_output=max_force).to(device)
        self.optim = optim.Adam(self.model.parameters(), lr = lr)

        self.target = deepcopy(self.model).to(device)
        self.losses = []
        self.update_size = []
        self.device = device

    def optimise_step(self, critic, state, episode):
        state_action = torch.cat((state.to(self.device), self.model(state.to(self.device))), dim=-1)
        
        loss = -critic.model(state_action.to(self.device)).mean()

        self.optim.zero_grad()
        loss.backward()
        # if episode % 5 == 0 and episode > 100:
        #     print([p.grad.view(-1) for p in self.model.parameters()])
        self.optim.step()

    def update_target(self, tau):
        for target_param, local_param in zip(self.target.parameters(), self.model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)