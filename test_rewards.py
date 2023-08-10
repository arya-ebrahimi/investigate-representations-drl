import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd
import glob

sns.set_style("whitegrid", {'axes.grid' : True,
                            'axes.edgecolor':'black'})

    
files = glob.glob('.rewards/no_aux/*')
rewards = []

max_len = 0

for file in files:
    with open(file, 'rb') as fp:
        t = pickle.load(fp)
        if max_len < len(t):
            max_len = len(t)
        rewards.append(t)
        fp.close()
        
fig = plt.figure()
plt.clf()

rewards_t = torch.zeros((max_len), dtype=torch.float)
for i in range(len(rewards)):
    tt = torch.ones((max_len), dtype=torch.float)
    r = torch.tensor(rewards[i], dtype=torch.float)
    tt[:r.shape[0]] = r
    
    rewards_t += tt
# sns.lineplot(data=rewards_t.numpy())
rewards_t = rewards_t / len(rewards)

means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
means = torch.cat((torch.zeros(99), means))

sns.lineplot(rewards_t.numpy(), alpha=0.5, label='true rewards averaged over 5 runs')
sns.lineplot(means.numpy(), label='averaged reward over 100 episodes')

plt.xlabel('Episode')
plt.ylabel('Reward')

plt.title("Successor Features")

plt.legend(
frameon=True, fancybox=True, loc="lower right")

sns.despine()
plt.tight_layout()
plt.show()
