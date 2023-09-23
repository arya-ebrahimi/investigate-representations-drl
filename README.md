# Investigating the Representations & Auxiliary Tasks in DeepRL
----
## A minimal version of [Investigating the Properties of Neural Network Representations in Reinforcement Learning](https://arxiv.org/abs/2203.15955)
----

My bachelor project report is available at [ResearchGate](https://www.researchgate.net/publication/373818471_Investigating_Representations_and_Auxiliary_Tasks_in_DeepRL).

<p align="middle" >
  <img src="figures/out.gif" title="Main Task" width="200" />
</p>

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

The hyperparameters and additional settigns, like auxiliary tasks, are available in config folder. After those are set, start the training process as follows:

```bash
python main.py
```


Refer to [my blog post](https://arya-ebrahimi.github.io/posts/fuzzy-tiling-activations/) for more information about FTA activation function.
