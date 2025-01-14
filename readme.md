# TODO
First fix single agent by:
- Start from fleet_size = 1 maybe?
- Get reasonable performance in sb3 or PyTorch from scratch
- Try different observation config `obs_config`
- Change reward system and reward values
- Implement image-based observation -> need to implement Conv nn for actor critic, and - Color inversion is done in env side... 

Second:
- Improve reward and other mentioned above for mult-agent setup.

# Multi-agent Setup:
Currently PPO for multi-agent is done using self-play. It means same actor, critic nw is used for each agent's observation. 
`train_multi.py`: Pytorch multi-agent training with self-play

For Image input based observation:
`img_obs = True` -> Only environment is ready, need to implement from PPO side

# Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

# Environment Code

The environment code is located in `env_isolate.py`. It supports both single and multi-agent modes.
For single agent set num_agent=1

```
obs_config = {'enemy_ships': 3, 'asteroids': 2, 'boosts': 1, 'coins': 1, 'bullets': 3}

```
## Observation Space:
it means for the observation we use coordinates:
3 closest enemy ship per ship in fleet= 3*4 -> 0 for single agent training -> Note: chosen so that 12 are unique unlike below ones that can be duplicate
2 closest asteroid per ship in fleet = 2*4
1 boost ,,                           = 1*4
1 coins ,,                           = 1*4
3 bullet ,,                          = 3*4

8+4+4+12 = 28 x,y coordinates = 56 values

For each ship its stats:
x,y, rotation, health, defense boost yes/no, attack boost yes/no = 6 values -> 24 values for a fleet

Total: 80 for single agent setup

For mult-agent, it will be 80*num_agents.

## Action Space:

Thrust, Rotation, Shoot (if +ve shoot else dont)


# Fleet Composition

Each fleet consists of four types of ships:

- **Supporter**: Heals within the perimeter and decreases damage taken (increased defense for allies).
- **Defender**: Acts as a tank.
- **Attacker**: Deals high attack damage and has a high attack rate.
- **Speedster**: Has high movement speed and attack speed.

# Boosts

Available boosts include:

- Increased defense
- Increased attack damage
- Health restoration

# Damage Objects

Types of damage objects:

- Asteroid
- Bullet

# Rendering

To render the GUI, set:

```python
render_mode = 'human'
```

If you encounter a pygame error, run the following command in the terminal:

```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

# Configuration

Training configurations are located in `config.yaml`.
Set num_agent>1 for multi-agent self-play

# Monitoring

To monitor training, run the following command before starting:

```bash
wandb login
```

# Training Scripts

- `train.py`: PyTorch training from scratch. - for num_agents == 1 only!!
- `sb_train.py`: Stable Baselines training.
- `train_multi.py`: Pytorch multi-agent training with self-play - for num_agents > 1

# Constants and Reward

Game object constants and reward values are defined in `constants.py`.
Reward calculation occur in ```env.step()```


