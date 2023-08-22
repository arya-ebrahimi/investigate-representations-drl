# Investigating the Representations & Auxiliary Tasks in DeepRL
----
After cloning the repository, use the `environment.yml` file to create the conda env with approprate packages.

```bash
conda env create --file environment.yml
conda activate repdrl
```

or give it your desired name:

```bash
conda env create --file environment.yml --name YourName
conda activate YourName
```

## Fuzzy Tiling Activations

FTA uses the same mechanism for binning as Tiling Activation. However, rather than sharp rises or falls, it employs ReLU-like functions for smoother transitions, resulting in non-zero derivatives. FTA introduces a parameter called $\eta$ to manage sparsity and the extent of smoothing. When $\eta = 0$, FTA becomes equivalent to Tiling Activation. A larger $\eta$ corresponds to a broader range of ReLU-like values.

<p>
  <img src="/assets/img/fta/fta3.png" alt="drawing" width="512"/>
  <em> FTA output for \(k=4\), and \(\eta=0.1\)</em>
</p>

<p>
  <img src="/assets/img/fta/fta2.png" alt="drawing" width="512"/>
  <em> FTA output for \(k=4\), and \(\eta=0.2\)</em>
</p>

As can be observed in these figures, a larger $\eta$ leads to a broader range of ReLU-like values on both sides of the flat region.

If you wish to change these values and better understand FTA, start by cloning the original PyTorch implementation of FTA from [this repository](https://github.com/hwang-ua/fta_pytorch_implementation/tree/main) or use the version I modified for GPU usage from [here](https://github.com/Arya-Ebrahimi/rl-playground/blob/main/Deep-Q-Learning/fta.py). After that you can generate the plots I illustrated above by using the following code:

```python
import torch
from fta import FTA
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False,
                            'axes.edgecolor':'black'})

activation = FTA(tiles=4, bound_low=0, bound_high=1, eta=0.1, input_dim=1, device='cpu')

fig, axs = plt.subplots(4)

l = []
for i in range (0, 101):
    x = torch.tensor(i/100)
    l.append(activation(x).squeeze().numpy())

l = np.array(l)

for i in range(l.shape[1]):
    axs[i].plot(np.linspace(0, 1, l.shape[0]), l[:,i], linewidth=2)
    if i < l.shape[1]-1:
        axs[i].axes.get_xaxis().set_ticks([])
    axs[i].set(ylabel='Bin '+str(i+1))

plt.show()
```

## Experiments
I have tested the FTA activation function in two settings: one with a simple DQN agent in the [Gymnasium Taxi environment](https://gymnasium.farama.org/environments/toy_text/taxi/) and another in a Maze environment I implemented, based on the environment introduced in [[5]](#5).

### Taxi environment

<p>
  <img src="/assets/img/fta/taxi.gif" alt="animation" width="500"/>
</p>

The code for this test is available in [this](https://github.com/Arya-Ebrahimi/rl-playground/tree/main/Deep-Q-Learning) repository. The results are as follows:

<p>
  <img src="/assets/img/fta/fta_taxi.png" alt="drawing" width="512"/>
  <em> Taxi env average rewards after 10000 episode by using FTA activation function.</em>
</p>

<p>
  <img src="/assets/img/fta/relu_taxi.png" alt="drawing" width="512"/>
  <em> Taxi env average rewards after 10000 episode by using ReLU activation function.</em>
</p>

As illustrated in the two figures above, FTA has achieved more stable, efficient, and faster learning in comparison with the ReLU activation function.

### Maze environment

<p>
  <img src="/assets/img/fta/maze.gif" alt="animation" width="300"/>
  <em> FTA output for \(k=4\), and \(\eta=0.2\)</em>
</p>

In my final bachelor project, I tried to rebuild the work introduced in [[5]](#5), so I implemented a similar environment to theirs and a DQN agent with Polyak updates, along with some auxiliary tasks. The repository is available [here](https://github.com/Arya-Ebrahimi/investigate-representations-drl). In a part of my experiments, I compared the usages of ReLU and FTA, as depicted in the following figure: (Note that the plots are averaged over 5 different runs and they are more accurate than the Taxi environment)

<p>
  <img src="/assets/img/fta/relu-fta-comparison.png" alt="drawing" width="600"/>
  <em> Maze env average rewards.</em>
</p>

The FTA results are shaded in blue, while the ReLU results are illustrated in red. It is obvious that utilizing FTA could improve training by providing a more reliable and faster learning process within fewer episodes. As illustrated in this figure, FTA achieved a hundred consecutive successful episodes in a smaller number of total episodes compared to the auxiliary tasks based on ReLU.


This was only a brief introduction to FTA, where I shared my experiments. I omitted the implementation details and focused on presenting the main ideas. If you are further interested, please refer to the main article [[4]](#4), and [[5]](#5) could also provide valuable insights. Hope this blog was helpful to you!






<p>
  <img src="/assets/img/fta/fta1.png" alt="drawing" width="512"/>
  <em> Tiling activation output for \(k=4\)</em>
</p>


In this example, the number of bins, defined as $k$, is set to 4. We passed numbers between 0 and 1 to the Tiling activation function, and as shown in this figure, for each value of $z$, the corresponding activation will be triggered.

$$
\begin{cases}
  0.00 \leq z < 0.25 &&& \text{bins}=\begin{bmatrix} 1&0&0&0 \end{bmatrix}\\
  0.25 \leq z < 0.50 &&& \text{bins}=\begin{bmatrix} 0&1&0&0 \end{bmatrix}\\
  0.50 \leq z < 0.75 &&& \text{bins}=\begin{bmatrix} 0&0&1&0 \end{bmatrix}\\
  0.75 \leq z < 1.00 &&& \text{bins}=\begin{bmatrix} 0&0&0&1 \end{bmatrix}\\
\end{cases}
$$

Although Tiling Activation successfully generates sparse outputs, it has zero derivatives almost everywhere. This fact motivates the design of a smoother version called Fuzzy Tiling Activations.

## Fuzzy Tiling Activations

FTA uses the same mechanism for binning as Tiling Activation. However, rather than sharp rises or falls, it employs ReLU-like functions for smoother transitions, resulting in non-zero derivatives. FTA introduces a parameter called $\eta$ to manage sparsity and the extent of smoothing. When $\eta = 0$, FTA becomes equivalent to Tiling Activation. A larger $\eta$ corresponds to a broader range of ReLU-like values.

<p>
  <img src="/assets/img/fta/fta3.png" alt="drawing" width="512"/>
  <em> FTA output for \(k=4\), and \(\eta=0.1\)</em>
</p>

<p>
  <img src="/assets/img/fta/fta2.png" alt="drawing" width="512"/>
  <em> FTA output for \(k=4\), and \(\eta=0.2\)</em>
</p>

As can be observed in these figures, a larger $\eta$ leads to a broader range of ReLU-like values on both sides of the flat region.

If you wish to change these values and better understand FTA, start by cloning the original PyTorch implementation of FTA from [this repository](https://github.com/hwang-ua/fta_pytorch_implementation/tree/main) or use the version I modified for GPU usage from [here](https://github.com/Arya-Ebrahimi/rl-playground/blob/main/Deep-Q-Learning/fta.py). After that you can generate the plots I illustrated above by using the following code:

```python
import torch
from fta import FTA
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False,
                            'axes.edgecolor':'black'})

activation = FTA(tiles=4, bound_low=0, bound_high=1, eta=0.1, input_dim=1, device='cpu')

fig, axs = plt.subplots(4)

l = []
for i in range (0, 101):
    x = torch.tensor(i/100)
    l.append(activation(x).squeeze().numpy())

l = np.array(l)

for i in range(l.shape[1]):
    axs[i].plot(np.linspace(0, 1, l.shape[0]), l[:,i], linewidth=2)
    if i < l.shape[1]-1:
        axs[i].axes.get_xaxis().set_ticks([])
    axs[i].set(ylabel='Bin '+str(i+1))

plt.show()
```

## Experiments
I have tested the FTA activation function in two settings: one with a simple DQN agent in the [Gymnasium Taxi environment](https://gymnasium.farama.org/environments/toy_text/taxi/) and another in a Maze environment I implemented, based on the environment introduced in [[5]](#5).

### Taxi environment

<p>
  <img src="/assets/img/fta/taxi.gif" alt="animation" width="500"/>
</p>

The code for this test is available in [this](https://github.com/Arya-Ebrahimi/rl-playground/tree/main/Deep-Q-Learning) repository. The results are as follows:

<p>
  <img src="/assets/img/fta/fta_taxi.png" alt="drawing" width="512"/>
  <em> Taxi env average rewards after 10000 episode by using FTA activation function.</em>
</p>

<p>
  <img src="/assets/img/fta/relu_taxi.png" alt="drawing" width="512"/>
  <em> Taxi env average rewards after 10000 episode by using ReLU activation function.</em>
</p>

As illustrated in the two figures above, FTA has achieved more stable, efficient, and faster learning in comparison with the ReLU activation function.

### Maze environment

<p>
  <img src="/assets/img/fta/maze.gif" alt="animation" width="300"/>
  <em> FTA output for \(k=4\), and \(\eta=0.2\)</em>
</p>

In my final bachelor project, I tried to rebuild the work introduced in [[5]](#5), so I implemented a similar environment to theirs and a DQN agent with Polyak updates, along with some auxiliary tasks. The repository is available [here](https://github.com/Arya-Ebrahimi/investigate-representations-drl). In a part of my experiments, I compared the usages of ReLU and FTA, as depicted in the following figure: (Note that the plots are averaged over 5 different runs and they are more accurate than the Taxi environment)

<p>
  <img src="/assets/img/fta/relu-fta-comparison.png" alt="drawing" width="600"/>
  <em> Maze env average rewards.</em>
</p>

The FTA results are shaded in blue, while the ReLU results are illustrated in red. It is obvious that utilizing FTA could improve training by providing a more reliable and faster learning process within fewer episodes. As illustrated in this figure, FTA achieved a hundred consecutive successful episodes in a smaller number of total episodes compared to the auxiliary tasks based on ReLU.


This was only a brief introduction to FTA, where I shared my experiments. I omitted the implementation details and focused on presenting the main ideas. If you are further interested, please refer to the main article [[4]](#4), and [[5]](#5) could also provide valuable insights. Hope this blog was helpful to you!
