import gym
import random
import numpy as np
from IPython.display import clear_output
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from baselines.common.atari_wrappers import make_atari, wrap_deepmind

# env = gym.make("Breakout-ram-v0")
env = make_atari("BreakoutNoFrameskip-v4")

# env = gym.make('Assault-v0')

# env.render()

env = wrap_deepmind(env, frame_stack=True, scale=True)
seed = 35
env.seed(seed)
n_actions = env.action_space.n

##Necessary parameters
e = 1.0
min_e = 0.1
max_e = 1.0
e_range = (max_e - min_e)
batch = 35
n_steps_epi = 1000
h, w, c = 250, 160, 3
gam = 0.99
epi = 10
past_a = []
past_s = []
past_next_s = []
past_r = []
past_complete = []
past_epi_r = []
current_r = 0
n_epi = 0
n_frame = 0

epi_random_frame = 50000
epi_greedy = 10000
max_buffer = 1000
update_actions_q = 4
update_actions_q_hat = 1000
loss = keras.losses.Huber()


def build_model():
    inp = layers.Input(shape=(84,84,4,))
    l1 = layers.Conv2D(32,10,strides=4,activation='relu')(inp)
    l2 = layers.Conv2D(64,5,strides=3,activation='relu')(l1)
    l3 = layers.Conv2D(128,3,strides=2,activation='relu')(l2)
    l4 = layers.Flatten()(l3)
    l5 = layers.Dense(512,activation='relu')(l4)
    l6 = layers.Dense(256,activation='relu')(l5)
    action = layers.Dense(n_actions, activation='linear')(l6)
    return keras.Model(inputs=inp,outputs=action)
    
Q = build_model()
Q_hat = build_model()
learning_rate = 0.0001
optim = keras.optimizers.Adam(learning_rate= learning_rate,clipnorm=1.0)

for i_episode in range(1, epi+1):
    state = np.array(env.reset())
    complete = False
    score = 0

    # while not complete: ##while the episode is not complete
    for i in range(1, n_steps_epi+1): 
        n_frame += 1

        if n_frame < epi_random_frame or e > np.random.rand(1)[0]:
            action  = np.random.choice(n_actions)
        else:
            s_t = tf.convert_to_tensor(state)
            s_t = tf.expand_dims(s_t, 0)
            a_prob = Q(s_t,training=False)
            action = tf.argmax(a_prob[0]).numpy()

        e -= e_range/epi_greedy
        e = max(e, min_e)
        next_s, r, complete, info = env.step(action)
        next_s = np.array(next_s)

        score += r
        past_a.append(action)
        past_s.append(state)
        past_next_s.append(next_s)
        past_complete.append(complete)
        past_r.append(r)
        state = next_s

        if n_frame % update_actions_q == 0 and len(past_complete) > batch:
            idx = np.random.choice(range(len(past_complete)), size=batch)

            sample_s = np.array([past_s[i] for i in idx])
            sample_next_s = np.array([past_next_s[i] for i in idx])
            sample_r = [past_r[i] for i in idx]
            sample_a = [past_a[i] for i in idx]
            sample_complete = tf.convert_to_tensor([float(past_complete[i]) for i in idx])
            
            f_r = Q_hat.predict(sample_next_s)
            
            updated_q = sample_r + gam *tf.reduce_max(f_r,axis=1)
            updated_q = updated_q * (1 - sample_complete) - sample_complete
            m = tf.one_hot(sample_a, n_actions)

            with tf.GradientTape() as tape:
                q_val = Q(sample_s)
                q_act = tf.reduce_sum(tf.multiply(q_val, m), axis=1)
                l = loss(updated_q, q_act)

            descents = tape.gradient(l, Q.trainable_variables)
            optim.apply_gradients(zip(descents, Q.trainable_variables))

        if n_frame % update_actions_q_hat == 0:
            Q_hat.set_weights(Q.get_weights())

        if len(past_r)>max_buffer:
            del past_r[:1]
            del past_s[:1]
            del past_next_s[:1]
            del past_a[:1]
            del past_complete[:1]

    past_epi_r.append(score)
    if len(past_epi_r) > 100:
        del past_epi_r[:1]
    current_r = np.mean(past_epi_r)

    n_epi += 1

    print("Episode is: {} and Score is: {}".format(epi, score))

env.reset()
for _ in range(1000):
    state = np.array(env.reset())
    s_t = tf.convert_to_tensor(state)
    s_t = tf.expand_dims(s_t, 0)
    a_prob = Q(s_t,training=False)
    action = tf.argmax(a_prob[0]).numpy()

    env.render()
    env.step(action) # take a random action
env.close()

