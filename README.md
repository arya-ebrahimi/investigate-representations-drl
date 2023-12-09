# Investigating the Representations & Auxiliary Tasks in DeepRL
----


The project's report is available at [ResearchGate](https://www.researchgate.net/publication/373818471_Investigating_Representations_and_Auxiliary_Tasks_in_DeepRL).

<p align="middle" >
  <img src="figures/out.gif" title="Main Task" width="200" />
</p>

After cloning the repository, use the `environment.yml` file to create the conda env with appropriate packages.

```bash
conda env create --file environment.yml
conda activate repdrl
```

or give it your desired name:

```bash
conda env create --file environment.yml --name YourName
conda activate YourName
```

#### tree of directories and files
```bash
├── config
│   ├── config.yaml
│   └── transfer_config.yaml
├── core
│   ├── activations
│   │   ├── fta.py
│   ├── agent.py
│   ├── envs
│   │   ├── __init__.py
│   │   └── MazEnv.py
│   ├── __init__.py
│   ├── nn.py
│   └── utils.py
├── environment.yml
├── figures
├── future
├── main.py
├── outputs
├── plot_rewards.py
├── README.md
├── test_env.py
├── test_learned_policy.py
├── test_reconstruction_aux.py
└── transfer.py

```

- config contains the configuration files for agents, such as nueral network hyperparameters and other settings of training process.
- core contains the main implementations, like FTA activation function, MazEnv customized environment, agent, nueral networks, and some supplementary functions in utils.
- environment.yml contains the dependencies and packages of conda environment.
- main.py is the main code to run for training the main task of agent.
- plot_rewards is used in order to create the plots used in the report file
- test_env is used to test the environment and its correctness.
- test_learned_policy is used to test a learned policy and creating gifs from its functionality.
- test_reconstruction_aux is used to test the output of reconstruction auxiliary task to see if its correct.
- transfer.py is the main training function of transfer tasks (after training on the main task, the weights are loaded for further transfer tasks)


### Environment

MazEnv is a deterministic environment meaning that the outcome of actions taken by the agent is entirely predictable and does not involve any randomness. The state transitions and resulting rewards are fixed and do not vary across different runs of the same action sequence. The agent will be rewarded with +1 if it reaches the goal state within a specified horizon defined in the configuration file. However, for each time step where the agent fails to reach the goal, it receives a reward of 0.


The MazEnv primarily consists of two main functions: reset and step. The reset function initializes a new maze environment, setting the agent's starting position randomly from available positions that are neither walls nor the goal. Subsequently, it returns an RGB image representing the current environment.
The step function manages the transitions within the environment. It takes an action selected by the agent, calculates the next state, and computes the reward received by the agent for performing this action. Afterward, it updates the RGB image of the current state to reflect the agent's movement to the next state. Finally, the function returns the RGB image and the corresponding reward.

### Agent

The agent comprises two primary networks. First, a representation network is implemented, which processes the input observation image through two convolutional layers, followed by a single fully connected layer consisting of 32 neurons. The activation function for this layer can be either FTA or ReLU. Therefore, the representation can have a shape of either $640 \times 1$ or $32 \times 1$ depending on the chosen activation function. For our study, we utilized FTA with parameters $k=20$, lower bound of $-2$, upper bound of $2$, and $\eta=0.4$.


Moreover, the representations undergo processing within a multi-layer perceptron referred to as the value network. This architecture comprises a duo of fully connected layers, each comprising 64 neurons. The output layer of this network uses four neurons, each signifying the value of an action. Therefore, the agent's operational process involves taking an image, which are passed to the representation network to shape its features. Subsequently, the value network estimates the value associated with each action, enabling the agent to determine the optimal action by selecting the output with the highest value using an argmax operation.

### Training
To start the training, run `python3 main.py`.

The first step involves training the base model, and to achieve this the main goal is used as the primary task for the base model. Five separate runs for six different settings, based on the utilization of auxiliary tasks, are applied for both FTA and ReLU activation functions to monitor the training rewards of each episode during the training phase. The figures and comparisons are available in the report.

The second step is to use the trained weights of main model for the transfer tasks. Two transfer tasks are defined, which one of them is close to the main tasks and the other more differs with the main task. The outcomes of transferring knowledge to train both of these methods is available in the report.

Also, you can refer to [my blog post](https://arya-ebrahimi.github.io/posts/fuzzy-tiling-activations/) for more information about the FTA activation function.
