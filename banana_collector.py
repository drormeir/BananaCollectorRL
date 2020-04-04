import math
import numpy as np
import os
import shutil
from ddqn_agent import ddqn_agent
from banana_env import banana_env

class banana_collector:
    def __init__(self, goal = 13.0):
        self.goal = goal
        
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
        eps               = 1.0                         # initialize epsilon
        solved            = False
        while env.improvement >= 0:
            state = env.reset(train_mode=True) # reset the environment
            done     = False
            while not done:
                action, accumulate_reward   = agent.act(state, eps)                   # select an action
                next_state, reward, done, _ = env.step(action, accumulate_reward)
                agent.step(state, action, reward, next_state, done)
                state      = next_state
            eps = max(eps_end, eps_decay*eps) # decrease epsilon
            end = "\n" if env.improvement > 0 else ""
            print('\rEpisode {}\tAverage Score: {:.2f}\t Stdev Score: {:.2f}'.format(env.ind_episode,\
                                                                                     env.curr_average,\
                                                                                     env.curr_stdev), end=end)
            if not solved and env.curr_average>=self.goal:
                print('\nEnvironment goal reached in {:d} episodes!\tAverage Score: {:.2f}'.format(env.ind_episode, env.curr_average))
                solved = True
            if env.improvement > 0:
                print('Saving checkpoint...')
                agent.save(output_filename)
                continue
            if env.ind_episode >= n_episodes:
                break
            if env.improvement < 0:
                if agent.reload_and_learning_rate_step(output_filename):
                    env.continue_improvement()
        print('\nNo more improvements. End of training.')
        return env.scores
    
    def test(self, env, agent, output_filename, n_episodes=100):
        print("Begin test on ",n_episodes," episodes...")
        env.reset_scores_window(n_episodes)
        agent.load(output_filename)
        actions_histogram = np.zeros(env.action_size)
        for i_episode in range(1, n_episodes+1):
            state = env.reset(train_mode=True) # reset the environment
            done  = False
            while not done:
                action, _                  = agent.act(state)     # select an action
                actions_histogram[action] += 1
                next_state, _, done, _     = env.step(action)
                state  = next_state
        print("Test score:  Composite={:.2f}\t Average={:.2f}\t Stdev={:.2f}".format(env.best_window_score,\
                                                                                     env.curr_average,\
                                                                                     env.curr_stdev))
        print("Actions histogram: ",actions_histogram / np.sum(actions_histogram))
        return env.best_window_score,env.curr_average, env.curr_stdev
    
    def fullrun(self, env, agent, output_filename, max_steps=300):
        env.reset_scores_window(None)
        state = env.reset(train_mode=False) # reset the environment
        agent.load(output_filename)
        for i_step in range(max_steps):
            action, _ = agent.act(state)     # select an action
            next_state, _, done, _ = env.step(action)
            if done:
                break
            state  = next_state
        print("Full run is over...")
        

    def multitrain_tune_LR(self, env, output_filename, lr_min_exp, lr_max_exp, num_train = 20):
        best_score   = 0
        best_average = 0
        best_stdev   = 0
        res_filename = ""
        best_param   = 0
        temp_dir     = "dqn_multitrain_temp"
        os.makedirs(temp_dir, exist_ok=True)
        for i_train in range(num_train):
            lr_exp                = lr_min_exp + (lr_max_exp-lr_min_exp)*i_train/float(num_train-1)
            agent                 = ddqn_agent(state_size=env.state_size, action_size=env.action_size,\
                                               learning_rate = math.pow(10,lr_exp))
            current_filename      = os.path.join(temp_dir,"multitrain_" + str(i_train) + "_" + output_filename)
            scores                = self.train(env=env, agent=agent, output_filename=current_filename)
            score, average, stdev = self.test(env=env,agent=agent,output_filename=current_filename)
            if score > best_score:
                best_param   = lr_exp
                best_score   = score
                best_average = average
                best_stdev   = stdev
                res_filename = current_filename
                best_scores  = scores
                
        shutil.rmtree(output_filename)
        shutil.move(res_filename, output_filename)
        shutil.rmtree(temp_dir,ignore_errors=True)
        
        print("multitrain_tune_LR results:")
        print("best composite score = ",best_score)
        print("best average         = ",best_average)
        print("best stdev           = ",best_stdev)
        print("best param           = ",best_param)
        return best_param, best_scores, best_score, best_average, best_stdev
    