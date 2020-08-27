# Playing Games With Deep Reinforcement Learning

This repository provides all the code I used for a project called "Playing Games With Deep Reinforcement Learning", described in a detailed blog post I wrote [here](https://ytbai.net/2020/08/26/playing-games-with-deep-reinforcement-learning/).

## Guide on Using Notebooks

```plots.ipynb``` --- notebook for making all the plots and figures in the blog post

```test.ipynb``` --- notebook for testing the random and static players discussed in the blog post

Results from notebooks can be obtained simply by running them from beginning to end without any input or modifications from the user.

## Model Architectures

Here I outline the neural network architectures used for this project. The precise architecture in each case can be easily read off from the corresponding PyTorch code.

```model_factory/q_models.py``` --- This script provides network architectures for the action value function used in Q-learning. Only the ```QValueWide``` model is used for this project.

```model_factory/ac_models.py``` --- This script provides architectures for joint models that predict a policy and state value function from the same network, used in both AC and PPO. In particular ```ACValueWide``` is used for the "Results" section of the blog post. In the section "Dependence on Width and Depth", the "baseline", 2x width, 4x width, +4 layers, and +8 layers architectures are given by ```ACValue```, ```ACValueWide```, ```ACValueVeryWide```, ```ACValueDeep```, and ```ACValueVeryDeep```, respectively.
