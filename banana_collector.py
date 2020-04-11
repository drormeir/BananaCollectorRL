import math
import numpy as np
import os
import shutil
from ddqn_agent import ddqn_agent
from banana_env import banana_env

class banana_collector:
    def __init__(self, goal = 13.0):
        self.goal = goal

    def multitrain_tune_LR(self, env, output_filename, lr_min_exp, lr_max_exp, num_train = 10):
        best_score   = 0
        best_average = 0
        best_stdev   = 0
        res_filename = ""
        best_param   = 0
        temp_dir     = "multitrain_tune_LR"
        shutil.rmtree(temp_dir,ignore_errors=True)
        os.makedirs(temp_dir)
        for i_train in range(num_train):
            lr_exp                = lr_min_exp + (lr_max_exp-lr_min_exp)*i_train/float(num_train-1)
            agent                 = ddqn_agent(state_size=env.state_size, action_size=env.action_size,\
                                               learning_rate = math.pow(10,lr_exp))
            current_filename      = os.path.join(temp_dir,"multitrain_" + str(i_train) + ".pth")
            scores, score, average, stdev = self.train(env=env, agent=agent, output_filename=current_filename)
            if score > best_score:
                best_param   = lr_exp
                best_score   = score
                best_average = average
                best_stdev   = stdev
                res_filename = current_filename
                best_scores  = scores
                
        shutil.rmtree(output_filename,ignore_errors=True)
        shutil.move(src=res_filename, dst=output_filename)
        shutil.rmtree(temp_dir,ignore_errors=True)
        
        print("multitrain_tune_LR results:")
        print("best composite score = {:.2f}",best_score)
        print("best average         = {:.2f}",best_average)
        print("best stdev           = {:.2f}",best_stdev)
        print("best param           = {:.3e}",best_param)
        return best_param, best_scores, best_score, best_average, best_stdev
        
    def train(self, env, agent, output_filename, n_episodes = 10000, window_size=100, eps_end=0.01, eps_decay=0.99):
        """Deep Q-Learning.

        Params
        ======
            env (banana_env): the environment
            agent (ddqn_agent): the agent
            output_filename (string): file name for check point
            n_episodes (int): maximum number of training episodes
            eps_end (float): minimum value of epsilon-greedy action selection
            eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        """
        env.reset_scores_window(window_size)
        eps  = 1.0                         # initialize epsilon
        goal = False
        temp_checkpoint_name = "temp_checkpoint.pth"
        while env.improvement >= 0:
            state = env.reset(train_mode=True) # reset the environment
            done     = False
            while not done:
                action, accumulate_reward   = agent.act(state, eps)                   # select an action
                next_state, reward, done, _ = env.step(action, accumulate_reward)
                agent.step(state, action, reward, next_state, done)
                state                       = next_state
            eps         = max(eps_end, eps_decay*eps) # decrease epsilon
            end         = "\n" if env.improvement > 0 else ""
            ind_episode = len(env.scores)
            print('\rEpisode {}\t {}'.format(ind_episode,env.str_curr_window_score()), end=end)
            if not goal and env.curr_window_average>=self.goal:
                print('\nEnvironment goal reached in {:d} episodes!'.format(ind_episode))
                goal = True
            if env.improvement > 0:
                print('Saving checkpoint...')
                agent.save(temp_checkpoint_name)
            if ind_episode >= n_episodes:
                # rare case: maximum number of episodes  -->  end loop in any case
                if self.__test_improvement(env, agent):
                    agent.save(output_filename)
                break
            if env.improvement >= 0:
                continue
            # env.improvement < 0   -->  environment signals no more improvements...
            # test if reduce learning rate can make things any better
            agent.load(temp_checkpoint_name)
            if not self.__test_improvement(env, agent):
                break
            agent.save(output_filename)
            if agent.is_lr_at_minimum():
                break
            agent.learning_rate_step()

        shutil.rmtree(temp_checkpoint_name,ignore_errors=True) # avoid file not found error
        print('\nNo more improvements. End of training.')
        return env.scores, env.best_test_score, env.best_test_average, env.best_test_stdev
    
    def test(self, env, agent, output_filename, n_episodes=100):
        print("Begin test on ",n_episodes," episodes...")
        env.reset_scores_window(n_episodes)
        agent.load(output_filename)
        actions_histogram = np.zeros(env.action_size)
        for i_episode in range(n_episodes):
            state = env.reset(train_mode=True) # reset the environment
            done  = False
            while not done:
                action, _                  = agent.act(state)     # select an action
                actions_histogram[action] += 1
                state, _, done, _          = env.step(action)
            print("\rEpisode: {} out of: {}".format(i_episode+1,n_episodes),end="")
        print("\nTest score: {}".format(env.str_curr_window_score()))
        print("Actions histogram: ",actions_histogram / np.sum(actions_histogram))
        return env.curr_window_score,env.curr_window_average, env.curr_window_stdev

    def fullrun(self, env, agent, output_filename, max_steps=300):
        env.reset_scores_window(0)
        state = env.reset(train_mode=False) # reset the environment
        agent.load(output_filename)
        for i_step in range(max_steps):
            action, _         = agent.act(state)     # select an action
            state, _, done, _ = env.step_without_update(action)
            if done:
                break
        print("Full run is over...")
        
    def __test_improvement(self, env, agent):
        print("\nCompare last checkpoint against best test score so far...");
        test_window_scores = []
        for i_episode in range(env.window_size):
            state = env.reset(train_mode=True) # reset the environment
            done  = False
            while not done:
                action, _         = agent.act(state)     # select an action
                state, _, done, _ = env.step_without_update(action)
            test_window_scores.append(env.score)
            print("\rEpisode: {} out of: {}".format(i_episode+1,env.window_size),end="")
        print("\n")
        return env.continue_improvement(test_window_scores)
    
