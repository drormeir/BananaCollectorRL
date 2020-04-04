import numpy as np
import os
import shutil
import torch
import torch.nn.functional as F
import torch.optim as optim

from replay_buffer import ReplayBuffer
from model import DuelQNetwork

class ddqn_agent():
    """Interacts with and learns from the environment."""
    def __init__(self, state_size, action_size,\
                 hidden_layers = [[64,64,32],[],[]],\
                 update_every  = 4,\
                 batch_size    = 128,\
                 buffer_size   = int(1e5),\
                 learning_rate = 5e-4,\
                 tau           = 1e-3,\
                 gamma         = 0.99,\
                 random_walk   = [0.75, 0.05, 0.1, 0.1],\
                 device        = None,\
                 verbose_level = 2):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int):  dimension of each state
            action_size (int):  dimension of each action
            hidden_layers (list of lists of ints): hidden layers structure
            update_every (int): how often to update the network
            batch_size (int):   minibatch size for sampling replay buffer and train the network
            learning_rate (float): learning rate of the local network
            tau(float): for soft update of target parameters
            gamma (float):  discount factor of next state Q value
            random_walk (array-like of action_size ints): apriory probabilities of random walk
            device: cpu or gpu
        """
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_size    = state_size
        self.action_size   = action_size
        self.action_range  = np.arange(self.action_size)
        self.seed          = 0
        self.update_every  = update_every
        self.batch_size    = batch_size
        assert type(learning_rate) is float
        self.lr_max        = learning_rate
        self.lr_min        = 1e-6
        self.lr_decay      = 0.5
        self.hidden_layers = hidden_layers
        self.tau0          = tau
        self.buffer_size   = buffer_size
        self.gamma         = gamma
        self.verbose_level = verbose_level
        self.set_random_walk_probabilities(random_walk)
        self.reset()
    
    def reset(self):
        np.random.seed(self.seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step         = 0
        self.t_last_update  = 0
        self.tau            = 1.0
        self.prev_prev_act  = -1
        self.prev_act       = -1
        self.lr             = self.lr_max
        self.lr_at_minimum  = False
        if self.verbose_level > 0:
            print("Reset DDQN agent with parameters:")
            print("state_size=",self.state_size)
            print("action_size=",self.action_size)
            print("hidden_layers=",self.hidden_layers)
            print("seed=",self.seed)
            print("update_every=",self.update_every)
            print("batch_size=",self.batch_size)
            print("buffer_size=",self.buffer_size)
            print("learning_rate=",self.lr)
            print("tau=",self.tau0)
            print("random_walk=",self.random_walk)
            print("gamma=",self.gamma)
            
        # Q-Network
        self.qnetwork_local  = DuelQNetwork(self.state_size, self.action_size, self.hidden_layers, self.seed).to(self.device)
        self.qnetwork_target = DuelQNetwork(self.state_size, self.action_size, self.hidden_layers, self.seed).to(self.device)
        self.optimizer       = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)
        self.__set_lr()
        # Replay memory
        self.memory          = ReplayBuffer(self.state_size, self.action_size, self.buffer_size, self.batch_size, self.seed)
    
    def set_random_walk_probabilities(self, random_walk = [0.75, 0.05, 0.1, 0.1]):
        if len(random_walk) == self.action_size:
            t = type(random_walk)
            if t == list:
                self.random_walk = np.array(random_walk) / np.sum(random_walk)
                return
            if t == np.ndarray:
                self.random_walk = random_walk / np.sum(random_walk)
                return
            raise
        # random_walk is a number between 0.0 to 1.0
        # 0.0 --> equal probability   1.0 --> only forward walk
        # changing random walk probability to favor forward action over the rest of actions
        equal_prob          = 1.0/self.action_size
        forward_prob        = random_walk*1.0 + (1.0-random_walk)*equal_prob
        other_prob          = (1.0-forward_prob)/(self.action_size-1)
        self.random_walk    = np.full(self.action_size,fill_value=other_prob)
        self.random_walk[0] = forward_prob
        
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        next_value_multiplier = self.gamma*(1-done)
        self.memory.add(state, action, reward, next_state, next_value_multiplier)

        # Learn every UPDATE_EVERY time steps.
        self.t_step += 1
        if self.t_step < self.t_last_update + self.update_every:
            return
        self.t_last_update = self.t_step
        experiences = self.memory.sample()
        if experiences is None:
            # If enough samples are available in memory, get random subset and learn
            return
        states, actions, rewards, next_states, next_value_multipliers = experiences
        states                 = torch.from_numpy(states).float().to(self.device)
        actions                = torch.from_numpy(actions).long().to(self.device)
        rewards                = torch.from_numpy(rewards).float().to(self.device)
        next_states            = torch.from_numpy(next_states).float().to(self.device)
        next_value_multipliers = torch.from_numpy(next_value_multipliers).float().to(self.device)

        self.qnetwork_target.train() # batch norm can update itself
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets      = rewards + next_value_multipliers * Q_targets_next

        # Get expected Q values from local model
        self.qnetwork_local.train()
        Q_expected     = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- soft update target network ------------------- #
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
        self.tau = min(max(self.tau / (1 + self.tau - self.tau0),self.tau0),self.tau)
        
    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        # Epsilon-greedy action selection
        exploitation = np.random.random() >= eps
        if exploitation:
            # exploit the agent knowledge
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local(state).cpu().data.numpy()
            new_act = np.argmax(action_values)
        else:
            # explore the environment using random move
            # try to employ a "common sense" in order to avoid redundant random moves
            redundant = -1
            if self.prev_act == 0 and self.prev_prev_act == 1:
                # prev step was forward, before that was backward --> inhibit backward
                redundant = 1
            elif self.prev_act == 1 and self.prev_prev_act == 0:
                # prev step was backward, before that was forward --> inhibit forward
                redundant = 0
            elif self.prev_act == 2 and self.prev_prev_act == 3:
                # prev step was left, before that was right --> inhibit right
                redundant = 3
            elif self.prev_act == 3 and self.prev_prev_act == 2:
                # prev step was right, before that was left --> inhibit left
                redundant = 2
            random_walk = self.random_walk
            if redundant >= 0:
                # take the redundant move probability and split it to the rest of possible actions
                redundant_prob         = random_walk[redundant]
                random_walk           += redundant_prob/(self.action_size-1)
                random_walk[redundant] = 0
            new_act = np.random.choice(self.action_range, p=random_walk)
        self.prev_prev_act = self.prev_act
        self.prev_act      = new_act
        return new_act, exploitation

    def save(self, filename):
        shutil.rmtree(filename,ignore_errors=True)
        os.makedirs(filename)
        torch.save(self.qnetwork_local.state_dict(),  os.path.join(filename,"local.pth"))
        torch.save(self.qnetwork_target.state_dict(), os.path.join(filename,"target.pth"))
        torch.save(self.optimizer.state_dict(),       os.path.join(filename,"optimizer.pth"))

    def load(self, filename):
        self.__basic_load(filename) # do not change self.lr
        self.__set_lr()

    def reload_and_learning_rate_step(self, filename):
        if self.lr_at_minimum:
            if self.verbose_level > 1:
                print("\nCannot reduce learning rate because it is already at the minimum:",self.lr)
            return False
        self.__basic_load(filename) # do not change self.lr
        self.lr *= self.lr_decay
        if self.lr <= self.lr_min:
            self.lr_at_minimum = True
            self.lr            = self.lr_min
        self.__set_lr()
        return True
        
    def __basic_load(self, filename):
        self.qnetwork_local.load_state_dict(torch.load(os.path.join(filename,"local.pth")))
        self.qnetwork_local.load_state_dict(torch.load(os.path.join(filename,"target.pth")))
        self.optimizer.load_state_dict(torch.load(os.path.join(filename,"optimizer.pth")))
        
    def __set_lr(self):
        if self.verbose_level > 1:
            print("\nChanging learning rate to:",self.lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        
