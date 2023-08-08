import pickle
import matplotlib.pyplot as plt
import torch

with open('/home/arya/arya/rl/investigate-representations-drl/.models/rewards/da.pkl', 'rb') as fp:
    b = pickle.load(fp)
    
rewards_t = torch.tensor(b, dtype=torch.float)



print(rewards_t.numpy().shape)
plt.plot(rewards_t.numpy())
plt.show()
