import numpy as np
from typing import Optional
import gymnasium as gym
from gymnasium import spaces

DOWN = 0
RIGHT = 1
UP = 2
LEFT = 3

SIZE = 15

class MazEnv(gym.Env):
    
    '''
    MAZENV
    Customized maze environment which is used in my experiments
    '''
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    

    def __init__(self, render_mode=None, size=SIZE, goal_mode=0, virtual_goal=0):
        
        '''
        inputs:
            -size: 
                size of the sides of our square grid
                default value is 15
            
            -goal_mode: 
                can be 3 different values
                0 for main goal task
                1 for similar transfer goal task
                2 for more dis-similar transfer goal task
            
            -virtual_goal:
                can be 3 different values
                0 means no auxiliary virtual value function task
                1 for when we have vf1 auxiliary task
                2 for vf5 auxiliary task
        '''
        
        
        self.virtual_goal = virtual_goal
        
        if goal_mode == 0:
            goal_pose = [9, 10]
        elif goal_mode == 1:
            goal_pose = [9, 11]
        elif goal_mode == 2:
            goal_pose = [1, 4]
            
        if self.virtual_goal==1:
            self.virtual_goals = [[7, 7]]
        if self.virtual_goal==2:
            self.virtual_goals = [[0, 0], [0, 14], [14, 0], [14, 14], [7, 7]]
            
        self.size = size  # The size of the square grid
        self.shape = (size, size)
        
        # set observation_space and action_space of environment
        # observation space is rgb image of our maze so it would be (size, size, 3)
        # we have 4 actions (UP, DOWN, LEFT, RIGHT), so action_space would be 4
        
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.size, self.size, 3), dtype=np.uint8)
        self.action_space = spaces.Discrete(4)
        
        
        self.goal_state = np.ravel_multi_index((goal_pose[0], goal_pose[1]), self.shape)
        self.goal_state_index = np.unravel_index(self.goal_state, self.shape)

        # DEFINE WHAT EACH ACTION WILL DO:
        
        self._action_to_direction = {
            0: np.array([1, 0]),    # DOWN
            1: np.array([0, 1]),    # RIGHT
            2: np.array([-1, 0]),   # UP
            3: np.array([0, -1]),   # LEFT
        }
        
        # CREATING THE WALLS
        self.walls = self._calculate_wall()
        
        
        # CALCULATE POSSIBLE STARTING POINTS
        # at each episode agent will spawn in a random index
        # here we store the pizels in which there is no wall or goal in which agent can be spawned
        
        self.possible_starting_states = []
        for i in range(self.size):
            for j in range(self.size):
                if self.walls[i, j]==0 and not(i == goal_pose[0] and j == goal_pose[1]):
                    self.possible_starting_states.append(np.ravel_multi_index((i, j), self.shape))
        
        
        # calculate transition proabilities
        self.P = {}
        for s in range(self.size**2):
            pos = np.unravel_index(s, self.shape)
            self.P[s] = {a: [] for a in range(self.action_space.n)}
            self.P[s][RIGHT] = self._calculate_transition_prob(pos, self._action_to_direction[RIGHT])
            self.P[s][UP] = self._calculate_transition_prob(pos, self._action_to_direction[UP])
            self.P[s][LEFT] = self._calculate_transition_prob(pos, self._action_to_direction[LEFT])
            self.P[s][DOWN] = self._calculate_transition_prob(pos, self._action_to_direction[DOWN])
        

        self.render_mode = render_mode
        

        self.window = None
        self.clock = None
        
    def _calculate_wall(self):
        
        '''
        CALCULATE WALLS
        this function creates a (size, size) array of zeros and changes the index of predefined walls
        '''
        walls = np.zeros(self.shape)
        walls[2, 0:6] = 1
        walls[2, 8:] = 1
        walls[2:6, 5] = 1
        walls[5, 2:7] = 1
        walls[5, 9:] = 1
        walls[8:12, 2] = 1
        walls[11, 2:6] = 1
        walls[8:, 5] = 1
        walls[8, 9:] = 1
        walls[8:12, 9] = 1
        walls[11, 9:12] = 1
        
        return walls
        
    
    def _limit_coordinate(self, pos):
        
        '''
        this finction checks if the current state is valid (not hit the wall or fall out of the grid)
        
        inputs:
            -pos: the new position of agent after taking action
        outputs:
            -True if the action is valid or False if its invalid
    
        '''

        x = pos[0]
        y = pos[1]
        
        if x < 0 or x >= self.size or y < 0 or y >= self.size or self.walls[x][y]==1:
            return False
        return True

    def _calculate_transition_prob(self, current_pos, action):
        '''
        transition probabilities
        since the env is stationary, all actions have the probability of 1.0
        reward for reaching the goal is 1 and otherwise is 0
        
        inputs:
            -current_pos: current position of agent
            -action: the taken action
            
        outputs:
            -transition probabilities which is 1.0
            -new state of agent after taking the action
            -reward which is 1 for reaching the goal and 0 otherwise
            
        '''
        
        new_pos = current_pos + action
        if not self._limit_coordinate(new_pos):
            new_pos = current_pos
            
        new_state = np.ravel_multi_index(new_pos, self.shape)
        
        if new_state == self.goal_state:
            return[1.0, new_state, 1.0, True]
        
        return [1.0, new_state, 0.0, False]
    
    def calculate_virtual_reward(self):
        
        '''
        this function checks if the agent has reached any virtual goals
        
        outputs:
        1.0 reward if it reached a virtual goal and 0.0 otherwise
        '''
        
        for i in self.virtual_goals:
            if self.current_state_index[0] == i[0] and self.current_state_index[1] == i[1]:
                return 1.0
        
        return 0.0

    def step(self, a): 
        
        '''
        this is the step function of gym.Env which handles the transitions and rewards
        
        inputs:
            -a: the action taken at a timestep
            
        outputs:
            -next-state as an image
            -reward of taking that action
            -terminate if reached goal
            -truncate which is False always and is handled in agent side
            -additional info which contains virtual reward in case of having vf aux
        '''
        
        prob, next_state, reward, terminate = self.P[self.s][a]   
        self.image[self.current_state_index[0], self.current_state_index[1], 1] = 255
        self.s = next_state
        self.current_state_index = np.unravel_index(self.s, self.shape)
        self.image[self.current_state_index[0], self.current_state_index[1], 1] = 0
        self.last_action = a
        virtual_reward=None
        if self.virtual_goal != 0:
            virtual_reward = self.calculate_virtual_reward()
        return(self.image, reward, terminate, False, {'virtual-reward': virtual_reward})
        
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        
        '''
        this function handles the reset of gim.Env 
        it initiates a new state and selects a random position for agent to start in it
        outputs:
            -image, which is the initial state of agent
            
        '''
        
        super().reset(seed=seed)
        
        self.image = np.ones((self.size, self.size, 3), dtype='uint8') * 255
        self.image[self.walls==1, 0] = 0
        self.image[self.goal_state_index[0], self.goal_state_index[1]] = np.array([0, 255, 0])

        self.s = np.random.choice(np.array(self.possible_starting_states))
        self.last_action = None
        self.current_state_index = np.unravel_index(self.s, self.shape)
        self.image[self.current_state_index[0], self.current_state_index[1], 1] = 0
            
        return self.image, {"prob": 1}
    
    def return_virtual_goals(self):
        image, _ = self.reset()

        for i in self.virtual_goals:
            image[i[0], i[1], 2] = 128
            
        return image
