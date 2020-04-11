[//]: # (Image References)
[image1]: https://github.com/drormeir/BananaCollector/blob/master/TrainedAgent.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png "Kernel"

# BananaCollector
This project was submitted as part of [Udacity's Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) and is similar to [UnityML "Food Collector"](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#food-collector)

The purpose of the project is to build and train a single agent that navigates and collecting bananas in a large square world.
![Trained Agent][image1]

This game is episodic, where each episode is consists of 300 steps. For each step, a reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. The goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas. The minimal requierment for success is to have an average score of at least 13.0 points in 100 consecutive episodes.

The agent is implemented in Python 3.6 + PyTorch. The algorithm is based on ["Double Duel Q-network"](https://arxiv.org/abs/1511.06581) with "Epsilon-Greedy policy" for environment exploration and an "Experience Replay Buffer" as a dynamic dataset to learn from.

Original git repo of this project can be found at:
https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation

# Installation
To set up your python environment to run the code in this repository, follow the instructions below:
1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	conda activate drlnd
	```
2. Install pytorch using conda:
```
conda install pytorch=0.4.0 -c pytorch
```
3. Clone this git repo
```bash
git clone git@github.com:drormeir/BananaCollector.git
cd BananaCollector
pip install .
```

4. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)


5. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

5. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. 

![Kernel][image2]


# Usage
The jupyter notebook `Navigation_Test.ipynb` imports all necessary dependencies and the python files of this project.

# Report
A detailed report describing the learning algorithm, along with ideas for future work can be found in `report.md`
