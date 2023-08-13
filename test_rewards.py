import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd
import glob
import numpy as np
sns.set_style("whitegrid", {'axes.grid' : True,
                            'axes.edgecolor':'black'})

fig = plt.figure(figsize=(16, 9))
plt.clf()
colors=['blue', 'red', 'green', 'yellow', 'magenta', 'cyan']

for index, path in enumerate(['no_aux', 'ir', 'rp', 'vvf1', 'vvf5', 'sf']):
    
    if path == 'no_aux':
        name = 'No AUX'
    elif path == 'ir':
        name = 'IR'
    elif path == 'rp':
        name = 'RP'
    elif path == 'vvf1':
        name = 'VVF-1'
    elif path == 'vvf5':
        name = 'VVF-5'
    elif path == 'sf':
        name = 'SF'
        
    files = glob.glob('.transfer_rewards/'+path+'/*')
    rewards = []

    max_len = 0

    for file in files:
        with open(file, 'rb') as fp:
            t = pickle.load(fp)
            if max_len < len(t):
                max_len = len(t)
            rewards.append(t)
            fp.close()
            


    rewards_t = torch.zeros((max_len), dtype=torch.float)
    for i in range(len(rewards)):
        tt = torch.ones((max_len), dtype=torch.float)
        r = torch.tensor(rewards[i], dtype=torch.float)
        tt[:r.shape[0]] = r
        
        rewards_t += tt
    # sns.lineplot(data=rewards_t.numpy())
    rewards_t = rewards_t / len(rewards)

    means = rewards_t.unfold(0, 50, 1).mean(1).view(-1)
    mins = rewards_t.unfold(0, 50, 1).min(1)[0]
    maxs = rewards_t.unfold(0, 50, 1).max(1)[0]

    means = torch.cat((torch.zeros(49), means))
    mins = torch.cat((torch.zeros(49), mins))
    maxs = torch.cat((torch.zeros(49), maxs))

    # print(mins)
    # sns.lineplot(rewards_t.numpy(), alpha=0.5, label='true rewards averaged over 5 runs')
    sns.lineplot(means.numpy(), label=name, color=colors[index])
    sns.lineplot(mins.numpy(), alpha=0.0)
    c = sns.lineplot(maxs.numpy(), alpha=0.0)
    line = c.get_lines()
    plt.fill_between(line[index*3+0].get_xdata(), line[index*3+1].get_ydata(), line[index*3+2].get_ydata(), color=colors[index], alpha=.15, label=name+' observed range')


plt.xlabel('Episode', fontsize=14)
plt.ylabel('Average Reward', fontsize=14)

plt.title("Average reward over 5 runs", fontsize=20)

plt.legend(
frameon=True, fancybox=True, loc="lower right", prop={'size': 18})

sns.despine()
plt.tight_layout()
plt.savefig('figures/train.png')
# plt.show()
