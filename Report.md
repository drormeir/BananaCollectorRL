[//]: # (Image References)
[image1]: https://github.com/drormeir/BananaCollector/blob/master/Network_Diagram.jpg "Network Architecture"

# Banana Collector Report
This project inherits its central idea from Udacity's solution for the DQN in:
https://github.com/udacity/deep-reinforcement-learning/tree/master/dqn/solution

## Source files description
### replay_buffer.py
This file contains the Replay Buffer needed for learning.
For each step in the episode, the agent stores a tuple of data into this "Round Robin" buffer. After a given number of steps, the agent pulls a sample from this buffer and uses it as a mini-batch for the deep learning network.

The buffer size is large (100K tuples); The agent, every four steps, samples a mini-batch with is 64 tuples from the Replay Buffer.

Please note the source of Udacity, contains this Replay Buffer with a `deque` object, but here, the replay buffer using `np.array` for the sake of performance.

### model.py
The network architecture is an implementation of Duel DQN, which is proven to give better results than the classic DQN.
Here is a diagram that describes the network in this project:
![Network Architecture][image1]

The class `DuelQNetwork` can have the entire network architecture as a parameter that is consists of 3 lists:
* The first list is the sizes of the layers before the "Duel Fork".
* The second list is the sizes of the layers before the State's Value neuron.
* The third list is the sizes of the layers before the Actions' Advantage layer.

### ddqn_agent.py
The class `ddqn_agent` contains the implementation of the RL agent with an Epsilon-Greedy policy. Each step, the agent decides whether to explore new states by selecting a random action or to exploit its current knowledge of the environment. In addition to Udacity's original agent provided in the DQN solution, this agent features the following:
* Decaying learning rate: The higher level class that uses this agent may change the learning rate by a predefined factor (0.5). The learning rate decreases when the agent cannot improve any further during the main training loop.
* A semi-random action exploration instead of the regular random: When exploring the environment, the agent learns very slowly because the random actions do not have a common sense. Therefore, the agent is just wondering around randomly. Accidently, from time to time, the random steps on a banana and the agent can learn rare useful data; hence many steps are useless. Therefore, the agent uses a biased random with different probabilities for each direction that mimics the probabilities of a trained agent. Walking forward is the preferred action, and this method gives a real change in the state vector together with its reward. Hence, resulting in better relevant data in the replay buffer.
* Memory for avoiding redundant random moves: If an agent goes to the right, and then to the left, it will be useless to choose the right direction again afterward. Hence, this simple memory mechanism skips those unnecessary steps only when the agent decides to use random exploration.

### banana_env.py
The `banana_env` class inherits from the `UnityEnvironment` class and encapsulates the brain's index and brain's name within for clarity.

The `step` member function gives a better API for the user to interact with the environment. It mimics the simple `step` function API of the `OpenAI-Gym` environment class.

This class also keep tracking after the whole scores graph and after the most recent scores window. This window score is a `deque` object with a maximum length of 100. The member function `__update_curr_window_score` calculates the average of the window, its standard deviation, and a composite score, which is equal to the average minus standard deviation. 

Moreover, this class compares each new completed episode's score against the previous average window score and determines its improvement status. This calculation is performed in the member function: `__add_score_into_statistics`, the resulted status has 3 possible values:
* A status of +1 means "training should continue": This module checks for two things: First, if the recently completed episode score is better than the previous average score. Second,  if the current window's composite score is better than the previous window's composite score. In case both are true, then the environment will signal that training should continue.
* A status of -1 means "training should stop": This situation is the result of two checks: First if the current composite score is worse than the best composite score so far. Second, the number of completed episodes since the last improvement is larger than the window size. In other words, it means that the agent is way after its peak performance.
* A status of 0 means: "training can continue": It is the most likely result; still, the training process can decide to stop after too many tries.
* A unique approach when two cases happen together: First, the current window composite score is smaller than the best window's composite score. Second, the current average is bigger than the best window's average. An improvement may be ahead, hence the index of the best composite score so far is moved to the current index. This approach gives the training process extra tries until it gives up.

Another significant difference from the classic DQN algorithm, when counting the score of the entire episode, only the exploitation steps are calculated without the random exploration steps. The environment class uses a particular correction factor called`score_correction_factor` for evaluating the episode's score by the exploitation moves density.

### banana_collector.py
This class contains the highest level functions of controlling the training process.

The `train` member function is responsible for training a given agent in a given environment. After each episode, it reduces the exploration epsilon and checks the environment improvement status.
* If the improvement status is positive, then it saves the current agent state.
* If the improvement status is negative, the collector reduces the learning rate and continues with the training process. This reduction happens until the minimal learning rate has reached or a given amount of training tries has passed.

The 'test` member function loads a saved agent state and conducts 100 executions (exploitation only) on the environment and saves their scores. After that, it prints the statistics of those tests.

The 'multitrain_tune_LR' executes trains on the environment with a different learning rate each time. This function saves the best result of all these trains.

The 'fullrun' member function is for displaying the trained agent performance with full graphics enabled.

## The training process
The training process was performed in the Jupyter notebook: `Navigation_Test.ipynb`
At the bottom of this notebook, one can find the score graph.

## Challenges and further discussion
There are several challenges when trying to estimate the current score of the agent:

* After each several steps, the training process performs a single learning phase with a mini-batch. It is impossible to have an exact measurement of the current score of the agent to understand if we passed beyond the peak performance. Hence, the environment class uses the episode's score as a proxy to the exact value.
* As mentioned before, each episode may have some random exploration steps that have positive or negative rewards. Those exploration steps do not influence the full episode score because of the correction factor.
* The training process should stop when the agent reaches its peak performance. However, the moving average lags behind the true unknown average score. The program can calculate the correct average score by performing a real test (the one in `test` member function). The real test process does not use random exploration moves and does not learn any new actions' values. The resources to make a real test after each episode are too scarce. Hence there is no reasonable solution besides the lagging moving average.

## Future work:
The replay buffer is simple and does not take into account the different importance of each learning step. This project might use in the future a better experience manager called "Priority Experience Replay" (PER) that takes in to account the value misses and prioritizing them accordingly.

Dror Meirovich
