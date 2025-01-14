import os
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from env_isolate import MultiAgentSpaceShooterEnv
import wandb
from wandb.integration.sb3 import WandbCallback
from omegaconf import OmegaConf
import numpy as np

def lr_schedule(progress_remaining):
    return 1e-4 * progress_remaining

def make_env():
    """
    Create the environment for multi-agent training.
    """
    env = MultiAgentSpaceShooterEnv(
        num_agents=config['num_agents'],
        fleet_size=config['fleet_size'],
        max_fps=120,
        asteroid_count=config['asteroid_count'],
        boost_count=config['boost_count'],
        coin_count=config['coin_count'],
        render_mode=config['mode'],
        obs_config=obs_config,
        img_obs=config['img_obs']
    )
    return Monitor(env)

if __name__ == "__main__":
    config = OmegaConf.load("config.yaml")
    print(config)

    run = wandb.init(
        project="space-shooter",
        entity='spider-r-d',
        mode="offline",
        monitor_gym=True,
        config=OmegaConf.to_container(config),
        notes=config.notes,
        sync_tensorboard=True
    )

    track_file_Type = [".py", ".yaml", ".md"]
    wandb.run.log_code(
        ".",
        include_fn=lambda path: (
            any([path.endswith(file_type) for file_type in track_file_Type]) and ("wandb" not in path)
        )
    )

    obs_config = OmegaConf.to_container(config.obs_config)

    # Create the environment
    env = DummyVecEnv([make_env])

    if "sb3" not in os.listdir():
        os.mkdir("sb3")

    model_file = f"sb3/{run.id}"
    device = "cuda" if th.cuda.is_available() else "cpu"

    # Initialize PPO with a shared policy
    model = PPO(
        'MlpPolicy',
        env,
        verbose=1,
        learning_rate=lr_schedule,
        n_steps=config['update_timestep'],
        batch_size=128,
        device=device,
        n_epochs=config['K_epochs'],
        gamma=config['gamma'],
        clip_range=config['eps_clip'],
        tensorboard_log=f"sb3/{run.id}"
    )

    # Train the model
    model.learn(
        total_timesteps=config['max_training_timesteps'],
        callback=WandbCallback(
            verbose=2,
            model_save_freq=config['save_model_freq'],
            model_save_path=model_file
        )
    )

    # Save the final model
    model.save(model_file)