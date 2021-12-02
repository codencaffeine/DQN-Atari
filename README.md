# Deep Reinforcement Learning on Atari Games

<!-- ![alt text](./img/breakout.png) -->
<img src="./img/breakout.png" alt="breakout" width="600"/>  
Source [1]

Table of contents
=================
<!--ts-->
   * [Abstract](#abstract)
   * [Architecture](#architecture)
   * [Implementation](#implementation)
   * [Evaluation](#evaluation)
   * [References](#references)
<!--       * [STDIN](#stdin)
      * [Local files](#local-files) -->

## Abstract
Reinforcement learning is a subset of Machine learnig and is seen as a decision-making task that has components of control theory as well.
The basic idea is based on  the psychocological and neurospecific behavior of how animals react to a certain environment. An "Agent" is similar in that sense and tries to make enough representation of the environment given to them, their goal being, learning from the past experiences and improving the new experiences. In this project, using a deep neural network, an agent is created who's name is "deep-Q-network" that effectively learns from its environment using the high dimension input images and deploying end-to-end reinforcement learning. The test is performed on the Atari 2600 games to visualise the agent's performance over the period of time in terms of scores.

--------------

## Architecture
The architecture of this project involves two models: 
1) Q CNN (A convolutional Neural Network similar to the one implemented in the paper referenced above for action-value function Q)
2) Q_hat CNN (similar model as Q CNN  for target action-value function Q_hat)

The CNN has total 6 layers:
1) 3 Convolutional 2D layers
2) 3 Dense layers
The final layer outputs "Action-values"( Being in a state s, if we take action a how much will be the total reward)

--------------

## Implementation

1) The implementation is done using the Keras API of tensor flow(for approximating the Q value_function).
2) OpenAI gym is used for creating the environment of different Atari games and getting the observation space and action space values.
3) The mathematical flow of this project is exactly like the one implemented in the paper which is as given below:
# Mathematical Reference from the paper
<img src="./img/algo.png" alt="breakout" width="600"/> 

### Reference paper:
[1] Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., & Riedmiller, M. (2013). Playing atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602.

--------------

## Evaluation

The evaluation is done based on the improvement in the score of earned by the agent playing different games in Atari2600 which is shown as below:

--------------

## References  
#### [1] Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., & Riedmiller, M. (2013). Playing atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602.
--------------
