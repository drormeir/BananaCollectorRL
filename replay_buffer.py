import numpy as np

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, state_size, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.state_size  = state_size
        self.action_size = action_size
        self.batch_size  = batch_size
        self.buffer_size = buffer_size
        self.seed        = np.random.seed(seed)
        self.current_len = 0
        
        self.states      = np.empty((self.buffer_size,self.state_size), dtype=np.float32)
        self.actions     = np.empty((self.buffer_size,1),               dtype=np.int32)
        self.rewards     = np.empty((self.buffer_size,1),               dtype=np.float32)
        self.next_states = np.empty((self.buffer_size,self.state_size), dtype=np.float32)
        self.next_value_multiplier = np.empty((self.buffer_size,1),               dtype=np.float32)
        
        self.res_states      = np.empty((self.batch_size,self.state_size), dtype=np.float32)
        self.res_actions     = np.empty((self.batch_size,1),               dtype=np.int32)
        self.res_rewards     = np.empty((self.batch_size,1),               dtype=np.float32)
        self.res_next_states = np.empty((self.batch_size,self.state_size), dtype=np.float32)
        self.res_next_value_multiplier       = np.empty((self.batch_size,1),               dtype=np.float32)
            
    def add(self, state, action, reward, next_state, next_value_multiplier):
        """Add a new experience to memory."""
        assert action >= 0
        assert action < self.action_size
        ind_pos                     = self.current_len % self.buffer_size
        self.current_len           += 1
        self.states[ind_pos,:]      = state
        self.actions[ind_pos][0]    = action
        self.rewards[ind_pos][0]    = reward
        self.next_states[ind_pos,:] = next_state
        self.next_value_multiplier[ind_pos][0]      = next_value_multiplier
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        curr_len = min(self.current_len,self.buffer_size)
        if curr_len < self.batch_size:
            return None
        
        indexes       =  np.random.choice(range(curr_len), size=self.batch_size,replace=False)
        
        self.res_states[:]      = self.states[indexes,:]
        self.res_actions[:]     = self.actions[indexes,:]
        self.res_rewards[:]     = self.rewards[indexes,:]
        self.res_next_states[:] = self.next_states[indexes,:]
        self.res_next_value_multiplier[:]       = self.next_value_multiplier[indexes,:]
        return (self.res_states, self.res_actions, self.res_rewards, self.res_next_states, self.res_next_value_multiplier)
