import os
import glob
import time
from datetime import datetime

import torch
import numpy as np

import tqdm

from ppo_multi import PPO
from env_isolate import MultiAgentSpaceShooterEnv
import wandb

from constants import SEED
from omegaconf import OmegaConf
torch.manual_seed(SEED)
np.random.seed(SEED)


################################### Training ###################################
def train():
    print("============================================================================================")

    ####### initialize environment hyperparameters ######
    env_name = "single_agent_space_shooter"
    config_path = "config.yaml"
    config = OmegaConf.load(config_path)

    has_continuous_action_space = config.has_continuous_action_space

    max_ep_len = config.max_ep_len
    max_training_timesteps = config.max_training_timesteps

    print_freq = config.print_freq
    log_freq = config.log_freq
    save_model_freq = config.save_model_freq

    action_std = config.action_std
    action_std_decay_rate = config.action_std_decay_rate
    min_action_std = config.min_action_std
    action_std_decay_freq = config.action_std_decay_freq
    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    update_timestep = config.update_timestep
    K_epochs = config.K_epochs

    eps_clip = config.eps_clip
    gamma = config.gamma

    lr_actor = config.lr_actor
    lr_critic = config.lr_critic

    random_seed = SEED
    #####################################################

    print("training environment name : " + env_name)

    # env = gym.make(env_name)
    ### number of enemy ships, asteroids, boosts, coins, and bullets to consider in observation

    print("============================================================================================")
    print("config : ", config)
    run = wandb.init(project="space-shooter", entity='spider-r-d', mode="offline", config=OmegaConf.to_container(config), notes=config.notes).id
    track_file_Type = [".py",".yaml", ".md"]
    wandb.run.log_code(".",include_fn=lambda path: (any([path.endswith(file_type) for file_type in track_file_Type])) and ("wandb" not in path))

    # run = "test"
    obs_config = OmegaConf.to_container(config.obs_config)
    env = MultiAgentSpaceShooterEnv(num_agents=config['num_agents'], fleet_size=config['fleet_size'], max_fps=120, asteroid_count=config['asteroid_count'], boost_count=config['boost_count'], coin_count=config['coin_count'], render_mode=config['mode'], obs_config=obs_config, img_obs=config['img_obs'])

    # state space dimension
    assert env.observation_space.shape[0] % config['num_agents'] == 0
    state_dim = env.observation_space.shape[0]//config['num_agents']

    # action space dimensiony
    assert env.action_space.shape[0] % config['num_agents'] == 0
    action_dim = env.action_space.shape[0]//config['num_agents']

    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)
    #####################################################

    ################### checkpointing ###################
    run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
          os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
          os.makedirs(directory)


    checkpoint_path = directory + "{}_PPO_{}_{}_{}.pth".format(run,env_name, random_seed, run_num_pretrained)
    print("save checkpoint path : " + checkpoint_path)
    #####################################################


    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print("--------------------------------------------------------------------------------------------")
    if has_continuous_action_space:
        print("Initializing a continuous action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("starting std of action distribution : ", action_std)
        print("decay rate of std of action distribution : ", action_std_decay_rate)
        print("minimum std of action distribution : ", min_action_std)
        print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
    else:
        print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    #####################################################

    print("============================================================================================")

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std, config['num_agents'])

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    log_f = open(log_f_name,"w+")
    log_f.write('episode,timestep,reward\n')

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    # training loop
    tqdm_e = tqdm.tqdm(range(1, max_training_timesteps + 1), desc='Training')
    while time_step <= max_training_timesteps:

        env.reset()
        state = env.get_obs()
        current_ep_reward = 0

        for t in range(1, max_ep_len+1):
            tqdm_e.update(1)

            # select action with policy
            action = ppo_agent.select_action(state)
            state, reward, each_agent_done, done, trunc, _ = env.step(action)

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(each_agent_done)
            
            ### if multi-agent environment, sum up rewards for all agents
            if config['num_agents'] > 1:
                reward = sum(reward)
                
            time_step +=1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # log in logging file
            if time_step % log_freq == 0:

                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                wandb.log({"episode": i_episode, "timestep": time_step, "average_reward": log_avg_reward})
                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:

                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))
                
                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            if done:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    log_f.close()
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':
    train()
    
    
    
    
    
    
    