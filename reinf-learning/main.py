#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 15:13:06 2020

@author: farismismar
"""

import os
import time
from colorama import Fore, Style
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from environment import radio_environment
from DQNLearningAgent import DQNLearningAgent as QLearner # Deep with GPU and CPU fallback
#from QLearningAgent import QLearningAgent as QLearner

MAX_EPISODES = 1000
MAX_TIME_STEPS = 15

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
 
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # My NVIDIA GTX 1080 Ti FE GPU

output = pd.DataFrame()
summary = pd.DataFrame()

os.chdir('/Users/farismismar/Desktop/deep')

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['axes.titlepad'] = 18
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['font.size'] = 12
plt.rcParams['text.latex.preamble'] = \
    r'\usepackage{amsmath}\usepackage{amssymb}'


def run_agent_q(env, plotting=True):
    global output, summary
   
    max_episodes_to_run = MAX_EPISODES # needed to ensure epsilon decays to min
    max_timesteps_per_episode = MAX_TIME_STEPS
    successful = False
    
    episode_successful = [] # a list to save the good episodes
    Q_values = []   
    losses = []
    
    agent.prepare_agent(env)
    
    max_episode = -1
    max_reward = -np.inf
    
    # Implement the Q-learning algorithm
    for episode_index in 1 + np.arange(max_episodes_to_run):
        observation = env.reset()
    
        print('Ep.        | eta | TS | Recv. SINR | BS Tx Pwr | Reward | Action')
        print('--'*43)
            
        (_, _, pt_serving, _) = observation

        action = agent.begin_episode(observation)           
        # Let us know how we did.
        print(f'{episode_index}/{max_episodes_to_run} | {agent.exploration_rate:.2f} | 0 | - dB | - W | 0 | {action}')
              
        total_reward = 0
        done = False
       
        episode_loss = []
        episode_q = []

        output_z = pd.DataFrame()
        
        for timestep_index in 1 + np.arange(max_timesteps_per_episode):
            # Take a step
            next_observation, reward, done, abort = env.step(action)
            (x_t, y_t, tx_power_t, bf_index_t) = next_observation
                        
            received_sinr = env.received_sinr_dB

            # Remember the previous state, action, reward, and done
            agent.remember(observation, action, reward, next_observation, done)
                           
            # Learn policy through replay.
            q, loss = agent.replay()
                      
            episode_loss.append(loss)
            episode_q.append(q)

            # make next_state the new current state for the next frame.
            observation = next_observation
            total_reward += reward

            successful = done and (total_reward > 0) and (abort == False)
            
            # Let us know how we did.
            print(f'{episode_index}/{max_episodes_to_run} | {agent.exploration_rate:.2f} | {timestep_index} | {received_sinr} dB | {tx_power_t} W | {total_reward} | {action} | ', end='')
    
            # Finished too soon?
            if done and timestep_index < max_timesteps_per_episode - 2:
                total_reward += env.reward_min
                done = False
                
            # Store the action, reward, and observation elements, done|aborted
            # for further postprocessing and plotting
            output_t = pd.Series([episode_index, timestep_index, x_t, y_t, received_sinr, tx_power_t, bf_index_t, action, total_reward, done, abort])
            output_z = output_z.append(output_t, ignore_index=True)
            
            if abort == True:
                print('ABORTED.')
                break
            else:
                print()
            
            # Update for the next time step
            action = agent.act(observation, total_reward)

        # Episode ends
        loss_z = np.mean(episode_loss)
        q_z = np.mean(episode_q)
        output_z.loc[:, 'Loss'] = loss_z
        output_z.loc[:, 'Q'] = q_z
        
        if (successful == True) and (abort == False):
            print(Fore.GREEN + 'SUCCESS.  ' + Style.RESET_ALL + 'Total reward = {}, average Q = {:.2f}, average loss = {:.2f}.\n'.format(total_reward, q_z, loss_z))
            episode_successful.append(episode_index)
            
            # Keep an eye on the best episode
            if (total_reward > max_reward):
                max_reward, max_episode = total_reward, episode_index
        else:
            reward = 0
            print(Fore.RED + 'FAILED TO REACH TARGET.  ' + Style.RESET_ALL + 'Total reward = {}, average Q = {:.2f}, average loss = {:.2f}.\n'.format(total_reward, q_z, loss_z))
        
        losses.append(loss_z)
        Q_values.append(q_z)
        output = pd.concat([output, output_z], axis=0)
                
        if np.isnan(loss_z) and np.isnan(q_z):
            print('FATAL.  Loss and Q are both NaN.  Try to re-run.  Aborting.')
            break
        
        if (q_z < -100) or (loss_z > 1e6):
            print('FATAL.  No learning happening due to extreme Q or loss values.  Try to re-run.  Aborting.')
            break
        
    if (len(episode_successful) == 0):
        print("Goal cannot be reached after {} episodes.  Try to increase maximum episodes or reduce target.".format(episode_index))
    else:
        print(f'Episode {max_episode}/{MAX_EPISODES} generated the highest reward {max_reward}.')

    output.columns = ['Episode', 'Time', 'UE x', 'UE y', 'UE SINR', 'BS TX Power', 'Beam Index', 'Action', 'Cumul. Reward', 'Done', 'Abort', 'Loss', 'Q']
    output.to_csv('output.csv', index=False)
    
    if plotting:
        summary = pd.DataFrame(data={'Episode': 1 + np.arange(episode_index),
                                     'Avg. Loss': losses,
                                     'Avg. Q': Q_values})
        plot_summary(summary)


def plot_summary(df):
    fig = plt.figure(figsize=(8,5))
    
    plot1, = plt.plot(df['Episode'], df['Avg. Q'], c='blue')
    plt.grid(which='both', linestyle='--')
    
    ax = fig.gca()    
    ax_sec = ax.twinx()
    plot2, = ax_sec.plot(df['Episode'], df['Avg. Loss'], lw=2, c='red')       
    ax.set_xlabel(r'Episode')
    ax.set_ylabel(r'$Q$')
    ax_sec.set_ylabel(r'$L$')    
    plt.legend([plot1, plot2], [r'Average $Q$', r'Average loss'],
               bbox_to_anchor=(0.1, 0.0, 0.80, 1), bbox_transform=fig.transFigure, 
               loc='lower center', ncol=3, mode="expand", borderaxespad=0.)
    
    plt.tight_layout()
    plt.savefig('output.pdf', format='pdf')
    plt.show()
    plt.close(fig)
    

seeds = np.arange(1).tolist()

for seed in seeds:
 
    env = radio_environment(random_state=seed)
    agent = QLearner(random_state=seed)
    start_time = time.time()
    run_agent_q(env)
    end_time = time.time()
    
    print('Simulation took {:.2f} minutes.'.format((end_time - start_time) / 60.))
    
########################################################################################
