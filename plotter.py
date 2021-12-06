import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple, deque

f = open("log4_best.yml")
all_lines = f.readlines()

scores = deque(maxlen=100) 
scores_info = []
for i in range(len(all_lines)):
    current_score = float(all_lines[i].split(" | ")[1].split(" ")[1])
    scores.append(current_score)
    scores_info.append([current_score, np.mean(scores), np.std(scores)])

print(np.array(scores_info).shape)
df = pd.DataFrame(scores_info,columns=['score','mean_score','var'])

fig = plt.figure(num=None,figsize=(10, 5))
ax = fig.add_subplot(111)
episode = np.arange(len(scores_info))
plt.plot(episode,df['mean_score'])
plt.fill_between(episode,df['mean_score'].add(df['var']),df['mean_score'].sub(df['var']),alpha=0.3)
plt.title("LunarLander-v2")
ax.legend(["Average reward"])
plt.ylabel('Reward')
plt.xlabel('# of Episodes')
plt.show()