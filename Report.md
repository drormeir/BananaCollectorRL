[//]: # (Image References)
[image1]: https://github.com/drormeir/BananaCollector/blob/master/Network_Diagram.jpg "Network Architecture"

# Banana Collector Report
The implementation for this project is based on the solution for the DQN in:
https://github.com/udacity/deep-reinforcement-learning/tree/master/dqn/solution

## Source Files
### replay_buffer.py
This file contains the Replay Buffer needed for learning.
For each step in the episode, the agent stores a tuple of data into this "Round Robin" buffer. After a given number of steps, the agent pulls a sample from this buffer and uses it as a mini-batch for the deep learning network.

The buffer size is large (100K tuples), the size of the mini-batch is 64 tuples which are sampled every 4 steps.

Please note that in the source of Udacity, this Replay Buffer is implemented with a "deque" object, while here I implemented it with np.array for the sake of performance.

### model.py
The network architecture is an implementation of Duel DQN, which is proven to give better results than the basic DQN.
Here's a diagram that describes the network in this project:
![Network Architecture][image1]

The class `DuelQNetwork` can have the entire network architecture as a parameter that is consists of 3 lists. The first list is the sizes of the layers before the "Duel Fork", the second list is the sizes of the layers before the State's Value neuron, and the third list is the sizes of the layers before the Actions' Advantage layer.

### ddqn_agent.py
class `ddqn_agent` contains the implementation of the RL agent. In addition to the basic agent provided for DQN solution, this agent has the following:

* Decaying learning rate: The higher level class that uses this module can inspect and change the learning rate by a predefined factor (0.5) It is modified when the agent cannot improve any further during the main training loop.
* A semi-random action exploration instead of the regular random: When using regular random for exploration the agent learns very slowly because the random actions don't have a "common sense" and the agent just wondering around in a random walk. Accidently, from time to time the random steps on a banana and the agent can learn rare useful data, hence many steps are useless. After investigating a "normal behavior" of a trained agent, I decided to use a biased random with different probabilities for each direction. Walking forward is the preferred action and it gives a real change in the state vector together with its reward, resulting in better relevant data in the replay buffer.
* Memory for avoiding redundant random moves: If an agent goes to the right, and then to the left, it will be useless to choose again the right direction afterward. I implemented a simple mechanism for avoiding those redundant steps, for the case that random exploration is used.

### banana_env.py
### banana_collector.py
