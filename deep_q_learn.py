import gym
from gym.wrappers import Monitor
from baselines.common.atari_wrappers import make_atari, wrap_deepmind

import torch
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import random
from collections import namedtuple, deque
import math
import time


env_name = "LunarLander-v2"
# env_name = "CartPole-v0"

env = gym.make(env_name).unwrapped
# env.seed(0)

# Parameters
EPS_THRESHOLD = 1.0
EPS_END = 0.05
EPS_DECAY = 0.999
ACTION_DIM = env.action_space.n
STATE_DIM = env.observation_space.shape[0]
BUFFER_SIZE = 100000
BATCH_SIZE = 64
NETWORK_UPDATE = 1      # TODO
TARGET_UPDATE = 1     # TODO
REPLAY_START = 10000
NO_EPISODES = 1800      # TODO
TARGET_SCORE = 200.0    # TODO for LunarLander
GAMMA = 0.99
TAU = 1e-3
LEARNING_RATE=5e-4
HIDDEN_UNITS = [64, 64]
TEST = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("GPU Available? [{}]".format(torch.cuda.is_available()))
total_steps = 0

class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer = deque([],maxlen=buffer_size)

    def store(self, *args):
        experience = namedtuple('Experience',
                        ('state', 'action', 'next_state', 'reward', 'done'))
        self.buffer.append(experience(*args))

    def getBatch(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers, seed):
        super(QNetwork, self).__init__()
        # self.seed = torch.manual_seed(seed)

        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, hidden_layers[0])])
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(hidden_layers[-1], action_size)

    def forward(self, state):
        x = state
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
        return self.output(x)


loss_log = []
Q_policy = QNetwork(state_size=STATE_DIM, action_size=ACTION_DIM, hidden_layers=HIDDEN_UNITS, seed=0).to(device)
Q_policy_target = QNetwork(state_size=STATE_DIM, action_size=ACTION_DIM, hidden_layers=HIDDEN_UNITS, seed=0).to(device)
optimizer = optim.Adam(Q_policy.parameters(), lr=LEARNING_RATE)

buffer = ReplayBuffer(BUFFER_SIZE)
reward_log = deque(maxlen=100) 

if not TEST:

    for episode in range(NO_EPISODES):
        state = env.reset()
        done = False
        episode_reward = 0
        frame = 0
        
        while not done:
            if random.random() > EPS_THRESHOLD:
                state_gpu = torch.from_numpy(state).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    action = np.argmax(Q_policy(state_gpu).cpu().data.numpy())
            else:
                action = random.choice(np.arange(ACTION_DIM))

            EPS_THRESHOLD = max(EPS_END, EPS_THRESHOLD*EPS_DECAY)

            next_state, reward, done, _ = env.step(action)
            buffer.store(state, action, next_state, reward, done)
            
            state = next_state
            episode_reward += reward
            frame += 1

            if len(buffer) > REPLAY_START:
                batch = buffer.getBatch(BATCH_SIZE)

                states = torch.from_numpy(np.vstack([b.state for b in batch if b is not None])).float().to(device)
                actions = torch.from_numpy(np.vstack([b.action for b in batch if b is not None])).long().to(device)
                next_states = torch.from_numpy(np.vstack([b.next_state for b in batch if b is not None])).float().to(device)
                rewards = torch.from_numpy(np.vstack([b.reward for b in batch if b is not None])).float().to(device)
                dones = torch.from_numpy(np.vstack([b.done for b in batch if b is not None]).astype(np.uint8)).float().to(device)

                Q_targets_next = Q_policy_target(next_states).detach().max(1)[0].unsqueeze(1)
                Q_targets = rewards + (GAMMA * Q_targets_next * (1 - dones))

                Q_expected = Q_policy(states).gather(1, actions)

                loss = F.mse_loss(Q_expected, Q_targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Soft update
                if frame % TARGET_UPDATE == 0:
                    for target_param, local_param in zip(Q_policy_target.parameters(), Q_policy.parameters()):
                        target_param.data.copy_(TAU*local_param.data + (1.0-TAU)*target_param.data)
        
        reward_log.append(episode_reward)    
        if episode % 1 == 0:
            print("Episode: {} | Score: {:.2f} | Average Score: {:.2f} | Epsilon: {}".format(episode, episode_reward, np.mean(reward_log), EPS_THRESHOLD))

        if episode % 100 == 0:
            timestr = time.strftime("%Y%m%d-%H%M%S")
            torch.save(Q_policy.state_dict(), 'models/%s_%s.pth'% (env_name, timestr))

        if np.mean(reward_log)>=TARGET_SCORE:
            print("Target Score Reached")
            timestr = time.strftime("%Y%m%d-%H%M%S")
            torch.save(Q_policy.state_dict(), 'models/%s_%s.pth'% (env_name, timestr))
            break

import glob
files = glob.glob("./models/*")
file_str = []
for i in range(len(files)):
    files[i] = files[i][:-4]
    file_str.append(files[i].split("/")[-1])

if TEST:

    for i in range(len(files)):
        filename = files[i]
        env = Monitor(env, './video/'+file_str[i], force=True)
        # print(filename)
        # filename = 'models/LunarLander-v2_20211206-113119'
        Q_policy.load_state_dict(torch.load('%s.pth'% (filename)))

        """ Test loop  """
        # for i_episode in range(1, 10+1):
        state = env.reset()
        score = 0
        done = False

        while not done:

            state_gpu = torch.from_numpy(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action = np.argmax(Q_policy(state_gpu).cpu().data.numpy())

            env.render()
            next_state, reward, done, _ = env.step(action)
            state = next_state
            score += reward

        env.close()
        print('\r#TEST Episode:{}, Score:{:.2f}'.format(0, score))

    """ End of the Test """
