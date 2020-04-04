from unityagents import UnityEnvironment
from collections import deque
import numpy as np
import math

class banana_env(UnityEnvironment):

    def __init__(self, file_name, ind_brain, no_graphics=False):
        super().__init__(file_name=file_name, no_graphics=no_graphics)
        self.ind_brain = ind_brain
        
        self.brain_name = super().brain_names[ind_brain]
        print("Selected brain name=",self.brain_name)

        self.brain = super().brains[self.brain_name]
        print("Selected brain=",self.brain)

        self.action_size = self.brain.vector_action_space_size
        print('Number of actions:', self.action_size)

        env_info = super().reset(train_mode=True)[self.brain_name] # reset the environment
        print('Number of agents:', len(env_info.agents))

        state = env_info.vector_observations[self.ind_brain]            # get the current state
        print('States look like:', state)
        self.state_size = len(state)
        print('States have length:', self.state_size)
        self.reset_scores_window(None)
        
    def reset_scores_window(self, window_size):
        if window_size is None:
            self.scores              = None
            self.window_size         = None
            self.scores_window       = None
            self.best_window_average = None
            self.best_window_stdev   = None
            self.best_window_score   = None
            self.best_window_episode = None
            self.temp_window_average = None
            self.temp_window_stdev   = None
            self.curr_average        = None
            self.curr_stdev          = None
            self.improvement         = None
        else:
            assert window_size > 1
            self.scores              = []                              # list containing scores from each episode
            self.window_size         = window_size
            self.scores_window       = deque(maxlen=self.window_size)  # last 100 scores
            self.best_window_average = 0
            self.best_window_stdev   = 0
            self.best_window_score   = -1e6
            self.best_window_episode = 0
            self.temp_window_average = 0
            self.temp_window_stdev   = 0
            self.curr_average        = 0
            self.curr_stdev          = 0
            self.improvement         = 0
        
    def reset(self, train_mode):
        env_info   = super().reset(train_mode=train_mode)[self.brain_name] # reset the environment
        state      = env_info.vector_observations[self.ind_brain]          # get the current state
        self.score = 0
        self.total_steps_in_episode = 0
        self.num_rewards_in_episode = 0
        return state
    
    def step(self, action, accumulate_reward=True):
        env_info    = super().step(int(action))[self.brain_name]   # send the action to the environment
        next_state  = env_info.vector_observations[self.ind_brain] # get the next state
        reward      = env_info.rewards[self.ind_brain]             # get the reward
        done        = env_info.local_done[self.ind_brain]          # see if episode has finished
        self.total_steps_in_episode += 1
        if accumulate_reward:
            self.num_rewards_in_episode += 1
            self.score                  += reward
        if done and self.scores is not None:
            if self.num_rewards_in_episode > 0:
                # score correction in case random step were used
                score_correction_factor = math.sqrt(self.total_steps_in_episode/float(self.num_rewards_in_episode))
                self.score *= score_correction_factor
            self.scores_window.append(self.score)       # save most recent score
            self.scores.append(self.score)              # save most recent score
            current_score_is_better_than_before = self.score > self.curr_average
            self.curr_average = np.mean(self.scores_window)
            self.curr_stdev   = np.std(self.scores_window,ddof=1) if len(self.scores_window) > 1 else np.inf
            curr_window_score = self.curr_average - self.curr_stdev
            self.improvement  = 0
            self.ind_episode  = len(self.scores)
            if self.ind_episode >= self.window_size:
                # estimate standard deviation of entire population
                if curr_window_score > self.best_window_score and current_score_is_better_than_before:
                    # real score improvement
                    self.improvement = 1
                    self.best_window_score   = curr_window_score
                    self.best_window_average = self.curr_average
                    self.best_window_stdev   = self.curr_stdev
                    self.temp_window_average = self.curr_average
                    self.temp_window_stdev   = self.curr_stdev
                    self.best_window_episode = self.ind_episode
                elif self.curr_average > self.temp_window_average:
                    # no real score improvement but still try a bit more...
                    self.temp_window_average = self.curr_average
                    self.best_window_episode = self.ind_episode                    
                elif self.curr_stdev < self.temp_window_stdev:
                    # no real score improvement but still try a bit more...
                    self.temp_window_stdev   = self.curr_stdev
                    self.best_window_episode = self.ind_episode                    
                elif self.ind_episode >= self.best_window_episode + self.window_size:
                    # no score improvement anymore
                    self.improvement = -1
        return next_state, reward, done, env_info
    
    def continue_improvement(self):
        self.improvement = 0
        #refresh scores_window for mean and stdev calculations
        ind_start_best_window = self.best_window_episode-self.window_size+1
        self.scores_window.extend(self.scores[ind_start_best_window:(self.best_window_episode+1)])
        self.temp_window_average = self.best_window_average
        self.temp_window_stdev   = self.best_window_stdev
        self.best_window_episode = self.ind_episode
        