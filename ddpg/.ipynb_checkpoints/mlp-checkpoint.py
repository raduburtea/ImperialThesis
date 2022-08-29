import torch
import torch.nn as nn
torch.manual_seed(12)

class MLP(nn.Module):
    
    def __init__(self, layer_tuple, output_activation=None, hidden_activation = nn.Tanh(), max_output=1):
        super(MLP, self).__init__()

        self.layers = nn.ModuleList([nn.Linear(layer_tuple[l], layer_tuple[l+1]) for l in range(len(layer_tuple)-1)])
        for layer in self.layers:
            nn.init.normal_(layer.weight, mean = 0, std = 1)
        self.hidden_activation = hidden_activation

        self.max_output = max_output
        self.output_activation = nn.Identity()
        if output_activation == 'relu':
            self.output_activation = nn.ReLU()
        if output_activation == 'tanh':
            self.output_activation = nn.Tanh()

    def forward(self, x):

        for layer in self.layers[:-1]:
            x = self.hidden_activation(layer(x))
        
        return self.max_output*self.output_activation(self.layers[-1](x))
