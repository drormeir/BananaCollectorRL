import torch
import torch.nn as nn
import torch.nn.functional as F
        
class DuelQNetwork(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, state_size, action_size, hidden_layers= [[64,64,32],[],[]], seed=0):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_layers (list of int): Number of nodes in hidden layers: common_list, value_list, actions_list
        """
        super().__init__()
        self.seed                = torch.manual_seed(seed)
        self.common_layers_list  = nn.ModuleList()
        self.value_layers_list   = nn.ModuleList()
        self.actions_layers_list = nn.ModuleList()
        # general configuration will be: states,b, L,b,r, L,b,r, L,b,r, actions
        self.common_layers_list.append(nn.BatchNorm1d(state_size))
        prev_size = state_size
        for hidden_size in hidden_layers[0]:
            self.common_layers_list.append(nn.Linear(in_features=prev_size,out_features=hidden_size))
            self.common_layers_list.append(nn.BatchNorm1d(hidden_size))
            # relu layer cannot be added here because relu is not module subclass
            prev_size = hidden_size
        prev_value = prev_size
        for hidden_size in hidden_layers[1]:
            self.value_layers_list.append(nn.Linear(in_features=prev_value,out_features=hidden_size))
            self.value_layers_list.append(nn.BatchNorm1d(hidden_size))
            # relu layer cannot be added here because relu is not module subclass
            prev_value = hidden_size
        self.value_layers_list.append(nn.Linear(in_features=prev_value,out_features=1))
        prev_action = prev_size
        for hidden_size in hidden_layers[2]:
            self.actions_layers_list.append(nn.Linear(in_features=prev_action,out_features=hidden_size))
            self.actions_layers_list.append(nn.BatchNorm1d(hidden_size))
            # relu layer cannot be added here because relu is not module subclass
            prev_action = hidden_size
        self.actions_layers_list.append(nn.Linear(in_features=prev_action,out_features=action_size))

    def forward(self, state):
        """Build a network that maps state -> action values."""
        common = state
        for ind, layer in enumerate(self.common_layers_list):
            common = layer(common)
            # relu layer after couple of module sub classes: linear,batchnorm,relu, linear,batchnorm,relu,...
            if ind > 0 and ind % 2 == 0:
                common = F.relu(common)
        value = common
        for ind, layer in enumerate(self.value_layers_list):
            value = layer(value)
            # relu layer after couple of module sub classes: linear,batchnorm,relu, linear,batchnorm,relu,...
            if ind > 0 and ind % 2 == 0:
                value = F.relu(value)
        actions = common
        for ind, layer in enumerate(self.actions_layers_list):
            actions = layer(actions)
            if ind > 0 and ind % 2 == 0:
                actions = F.relu(actions)
        return actions - actions.mean().expand_as(actions) + value.expand_as(actions)


