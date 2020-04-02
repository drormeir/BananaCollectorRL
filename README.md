[//]: # (Image References)
[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# BananaCollector
This project was submitted as part of [Udacity's Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) and is based on [UnityML "Food Collector"](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#food-collector)

The purpose of the project is to build and train a single agant that navigates and collecting bananas in a large square world.
![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The agent is implemented in Python 3.6. The algorithm is based on Double Duel Q-network with Epsilon-Greedy policy and an Experience-Replay-Buffer. The neural networks are implemented with Pytorch.




