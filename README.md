# PyAgents: a library of DRL algorithms

## Agents

Currently, the following algorithms have been implemented:

* [REINFORCE: Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/content/pdf/10.1007/BF00992696.pdf)
* [A2C: Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)
* [PPO: Proximal Policy Optimization](https://arxiv.org/pdf/1707.06347.pdf)
* [DQN: Human level control through deep reinforcement learning (Mnih et al., 2015)](https://deepmind.com/research/dqn/)
  * [Double DQN: Deep Reinforcement Learning with Double Q-learning (van Hasselt et al., 2015)](https://arxiv.org/abs/1509.06461)
  * [Dueling DQN: Dueling Network Architectures for Deep Reinforcement Learning (Wang et al., 2015)](https://arxiv.org/abs/1511.06581)
* [DDPG: Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)
* [SAC: Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905)

* [PER: Prioritized Experience Replay (Schaul et al., 2016)](https://arxiv.org/abs/1511.05952)

## Usage
A demo script `train.py` is provided to train the implemented agents on some classical RL environments. Hyperparameters can be customized by using [gin configuration files](https://github.com/google/gin-config), as described below.

```
python train.py [-h] [-a AGENT] [-e ENV] [-n NUM_ENVS] [-c CONFIG_DIR] 
               [-o OUTPUT_DIR] [-tv TEST_VER] 
               [-s SEED] [--video | --no-video]
```

In the folder gins/sac are present the configuration files for my experiments. To test run:
```
python train.py -a sac -c gins/sac 
```

Command line arguments:

| Argument                                         | Description                                                                  |
| ------------------------------------------------ | -----------------------------------------------------------------------------|
| `-h, --help`                                     | Shows help message                                                           |
| `-a AGENT, --agent AGENT`                        | Algorithm to train (see choices above)                                       |
| `-e ENV, --env ENV`                              | Environment to use (see available ones in `train.py`)                        |
| `-n NUM_ENVS, --num-envs NUM_ENVS`               | Number of parallel training envs (vectorized environments)                   |
| `-c CONFIG_DIR, --config-dir CONFIG_DIR`         | Path to directory containing .gin config files (see `gins` for examples)     |
| `-o OUTPUT_DIR, --output-dir OUTPUT_DIR`         | Either directory to save models (training) or directory to load models from (testing)|
| `-tv TEST_VER, --test-ver TEST_VER`              | If provided, load and test version `tv` of the agent in directory `OUTPUT_DIR`|
| `-s SEED, --seed SEED`                           | (default: 42) random seed                                                    |
| `--video, --no-video`                            | (default: False) if True, record testing video every now and then            |