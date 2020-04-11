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
        self.scores              = []                              # list containing scores from each episode
        self.window_size         = window_size
        self.scores_window       = deque(maxlen=self.window_size)  # last 100 scores
        self.best_window_average = -np.inf
        self.best_window_stdev   = np.inf
        self.best_window_score   = -np.inf
        self.best_window_episode = 0
        self.temp_window_average = -np.inf
        self.curr_window_average = -np.inf
        self.curr_window_stdev   = np.inf
        self.best_test_score     = -np.inf
        self.best_test_average   = -np.inf
        self.best_test_stdev     = np.inf
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
        if done:
            if self.num_rewards_in_episode > 0:
                # score correction in case random step were used
                score_correction_factor = math.sqrt(self.total_steps_in_episode/float(self.num_rewards_in_episode))
                self.score *= score_correction_factor
            self.__add_score_into_statistics()
        return next_state, reward, done, env_info
    
    def step_without_update(self, action):
        env_info    = super().step(int(action))[self.brain_name]   # send the action to the environment
        next_state  = env_info.vector_observations[self.ind_brain] # get the next state
        reward      = env_info.rewards[self.ind_brain]             # get the reward
        done        = env_info.local_done[self.ind_brain]          # see if episode has finished
        self.score += reward
        return next_state, reward, done, env_info
        
    def continue_improvement(self, test_window_scores):
        self.scores_window.extend(test_window_scores)
        self.__update_curr_window_score()
        if self.curr_window_score <= self.best_test_score:
            print("Current test score {:5.2f} is worse than previous test score {:5.2f}...".format(self.curr_window_score,\
                                                                                                   self.best_test_score))
            return False
        print("Current test score {:5.2f} is better than previous test score {:5.2f}...".format(self.curr_window_score,\
                                                                                                self.best_test_score))        
        self.best_test_score   = self.curr_window_score
        self.best_test_average = self.curr_window_average
        self.best_test_stdev   = self.curr_window_stdev
        self.__set_curr_window_as_best()
        self.improvement = 0 # train may continue
        return True
    
    def str_curr_window_score(self):
        return "Composite={:5.2f}\t Average={:5.2f}\t Stdev={:5.2f}".format(self.curr_window_score,\
                                                                         self.curr_window_average,\
                                                                         self.curr_window_stdev)

    def __add_score_into_statistics(self):
        if self.window_size is None or self.window_size < 2:
            return
        current_score_is_better_than_before = self.score > self.curr_window_average
        self.scores_window.append(self.score)       # save most recent score
        self.scores.append(self.score)              # save most recent score
        self.__update_curr_window_score()
        self.improvement  = 0 # train may continue
        num_episodes  = len(self.scores)
        if num_episodes < self.window_size:
            return
        if self.curr_window_score > self.best_window_score and current_score_is_better_than_before:
            # real score improvement :-)
            self.improvement = 1 # train should continue
            self.__set_curr_window_as_best()
        elif self.curr_window_average > self.temp_window_average:
            # no real score improvement but still try a bit more...
            self.temp_window_average = self.curr_window_average
            self.best_window_episode = num_episodes
        elif num_episodes >= self.best_window_episode + self.window_size:
            # no score improvement anymore :-(
            self.improvement = -1 # train should stop
        
    def __update_curr_window_score(self):
        self.curr_window_average = np.mean(self.scores_window)
        # estimate standard deviation of entire population
        self.curr_window_stdev   = np.std(self.scores_window,ddof=1) if len(self.scores_window) > 1 else np.inf
        self.curr_window_score   = self.curr_window_average - self.curr_window_stdev

    def __set_curr_window_as_best(self):
        self.best_window_score   = self.curr_window_score
        self.best_window_average = self.curr_window_average
        self.best_window_stdev   = self.curr_window_stdev
        self.temp_window_average = self.curr_window_average
        self.best_window_episode = len(self.scores)
        