# Deep Reinforcement Learning on Atari Games

<!-- ![alt text](./img/breakout.png) -->
<img src="./img/lunar_aishwarya_anilkumar.png" alt="breakout" width="600"/>  
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

<img src="./img/rein.png" alt="breakout" width="600"/> 
Image reference: Galatzer-Levy, Isaac & Ruggles, Kelly & Chen, Zhe. (2018). Data Science in the Research Domain Criteria Era: Relevance of Machine Learning to the Study of Stress Pathology, Recovery, and Resilience. 


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
## Mathematical Reference from the paper
<img src="./img/algo.png" alt="breakout" width="600"/> 

--------------

## Description

1) The main idea of this project is to combine the end-to-end reinforcement learning with deep neural network to train the D-Q-N agent.
2) The two neural networks implemented in this project are used to approximate the q tables for action-value function Q and target action_value function Q_hat
3) Q_hat is the initial policy, which Q is the current policy and gets updated and evaluates the initial policy Q_hat over many iterations.
4) in this program, I am updating the initial policy Q_hat every C steps making it equal to Q
5) The current policy Q now again starts updating based on updated Q_hat.
6) This process goes on for many episodes.
7) An episode is the time steps required by the agent from the initial state to the final state (final state being success or failure)
8) Now the process begins when the agent will make an action based on some initial policy. The agent starts from an initial state S1. This state is acquired by resetting the gaming environment (example: Atatri breakout)
9) Now the agent faces a dilemna: 
### "" Should I just always follow a certain policy and keep on exploiting to gain maximum reward? Or  should I explore more and see whether there are any better policies? ""
10) Good approach to solve this dilemna: To follow a some policy, but also have some probability for exploration.
Over the period of time, as the agent keeps learning, and tries to converge to near optimal policy, we will decrease this probability of choosing random action, and we will choose actions that are sampled from our learned policy
11) This makes the agent greedy over time
12) After we acquire an action from the previous step, we will use it in the gym environment and get the transition( current state, next state, reward, done)
13) We will use a replay memory to store all the previous n transitions. For training our agent we will randomly sample a fixed sized batch from this memory.
14) Now that we have sampled some past experiences, we will use our target policy Q_hat to calculate reward for all the sampled current states sj
We select a maximum action value from next state
If it’s a terminal state, the future reward = 0 which makes it just rj
15) With this new reward we just calculated for all sj, we will update our current policy Q network, with the new action values using gradient descent and back propogation
16) In my implementation I am updating the current network Q every 4 timesteps and the target network Q_hat every C timesteps
This delayed update is mainly because, the current Q network value does not change drastically over consecutive iterations.








--------------


## Evaluation

The evaluation is done based on the improvement in the score of earned by the agent playing different games in Atari2600 which is shown as below:
### Initial performance(Bad)
<img src="./img/breakout_bad.gif" alt="breakout" width="200"/> 


--------------


## Conclusion
Therefor, updating the current Q network can be seen as policy evaluation step in reinforcement learning
And updating the target Q_hat network which is my target network policy can be seen as policy improvement step of reinforcement learning.


--------------

## References  
#### [1] Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., & Riedmiller, M. (2013). Playing atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602.
--------------
