import gym
import random
import numpy as np
from IPython.display import clear_output
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from baselines.common.atari_wrappers import make_atari, wrap_deepmind

env = gym.make("Breakout-ram-v0")
# env = gym.make('Assault-v0')

env.render()

env = wrap_deepmind(env, frame_stack=True, scale=True)
seed = 35
env.seed(seed)
n_actions = env.action_space.n
e = 1.0
min_e = 0.1
max_e = 1.0
e_range = (max_e - min_e)
batch = 35
n_steps_epi = 1000
h, w, c = 250, 160, 3
gam = 0.99
epi = 10

for i_episode in range(1, epi+1):
    state = env.reset()
    complete = False
    score = 0
    while not complete:
        # env.render()
        # print(obs.shape)
        action = env.action_space.sample()
        n_state, reward, complete, info = env.step(action)
        score += reward

    print("Episode is: {} and Score is: {}".format(epi, score))
env.close()
h, w, c = n_state.shape[0], n_state.shape[1], n_state.shape[2]
print("h:{}, w:{}, c:{}, action space {}".format(h, w, c, n_actions))

print(env.observation_space.n)

def build_model():
    inp = layers.Input(shape=(84,84,4))
    l1 = layers.Conv2D(32,10,strides=4,activation='relu')(inp)
    l2 = layers.Conv2D(64,5,strides=3,activation='relu')(l1)
    l3 = layers.Conv2D(64,3,strides=2,activation='relu')(l2)
    l4 = layers.Flatten()(l3)
    l5 = layers.Dense(512,activation='relu')(l4)
    action = layers.Dense(n_actions, activation='linear')(l5)
    return keras.Model(inputs=inp,outputs=action)
    
model = build_model(h,w,c,actions=n_actions)
model = models.vgg16()
print(model)

## Here comes the Agent

# q_table = np.zeros([env.observation_space.n, env.action_space.n])