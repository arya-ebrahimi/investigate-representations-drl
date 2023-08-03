import pickle

with open('/home/arya/Desktop/investigate-representations-drl/.models/rewards/rewards_2023-08-03_09:10.pkl', 'rb') as fp:
    b = pickle.load(fp)
    
print(len(b))