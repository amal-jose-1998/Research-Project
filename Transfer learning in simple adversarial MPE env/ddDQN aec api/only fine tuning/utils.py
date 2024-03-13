import gym
import torch
import time
import argparse
import numpy as np
import wandb

def evaluate_policy(s, eval_env, agent_models, agent_id, write, agents, num_episodes=10):
    rewards = {}
    total_reward = {}
    for _ in range(num_episodes):
        eval_env.reset(seed = s+1)
        for agent_name in eval_env.agents:
            rewards[agent_name] = 0
        for agent in eval_env.agent_iter():
            id = agent_id[agent]
            observation, reward, termination, truncation, info = eval_env.last()
            if termination or truncation:
                a = None
            else:
                rewards[agent] += reward
                model = agent_models[id]
                a = model.select_action(torch.tensor(observation), evaluate=True)
            eval_env.step(a)

        for agent_name in agents:
            if agent_name not in total_reward:
                total_reward[agent_name] = 0.0
            total_reward[agent_name] += rewards[agent_name]
    
    for agent_name in agents:
        total_reward[agent_name] = total_reward[agent_name] / num_episodes
        if write:
            wandb.log({f'avg_reward on evaluation for {agent_name}': total_reward[agent_name]})
            print(f'avg_reward on evaluation for {agent_name}: ', total_reward[agent_name])
    return total_reward

def check_done_flag(termination, truncation):
    if termination or truncation:
        return True

def reset_episode_rewards(agents, total_game_reward):
    for s in agents:
        total_game_reward[s] = 0

def reset_epoch_rewards(agents, epoch_reward):
    for s in agents:
        epoch_reward[s] = 0

def reset_iter_counters(agents, total_training_steps, total_steps):
    for s in agents:
        total_training_steps[s] = 0
        total_steps[s] = 0

def loop_iteration(num_games, env, eval_env, opt, agent_models, agent_buffers, agents, agent_id, epochs, rand_seed): 
    schedualer = LinearSchedule(schedule_timesteps=opt.anneal_frac, final_p=0.02, initial_p=opt.exp_noise) #exploration noise linearly annealed from 1.0 to 0.02 within 200k steps
    loss = {}
    total_training_steps = {}
    total_steps = {} 
    total_game_reward = {}
    epoch_reward = {}
    # initialise the the counters for learning
    reset_iter_counters(agents, total_training_steps, total_steps)
    # initialise the the rewards for each agent
    reset_episode_rewards(agents, total_game_reward)
    reset_epoch_rewards(agents, epoch_reward)

    for i in range(num_games):
        print('episode:', i+1)
        wandb.log({'Episode': i+1})
        reset_episode_rewards(agents, total_game_reward)
    
        for _ in range(epochs): 
            print('epoch:', _+1)
            wandb.log({'Epoch': _+1})
            reset_epoch_rewards(agents, epoch_reward)
            env.reset(seed=rand_seed)
            starting_point ={}
            next_state = {}
            current_state = {}
            action = {}
            reward = {}
            done = {}
            for agent_name in env.agents:
                starting_point[agent_name] = True
                done[agent_name] = 0
            for agent in env.agent_iter():
                total_steps[agent]+=1
                wandb.log({f'total_steps {agent}': total_steps[agent]})
                observation, r, termination, truncation, info = env.last()
                id = agent_id[agent]
                if not starting_point[agent]:
                    next_state[agent] = observation
                    reward[agent] = r
                    wandb.log({f'rewards for {agent} at each step': r})
                    epoch_reward[agent]+=r
                    if termination or truncation:
                        done[agent] = 1
                    else:
                        done[agent] = 0
                    buffer = agent_buffers[id]
                    # convert to tensor
                    current_state[agent] = torch.tensor(current_state[agent])
                    action[agent] = torch.tensor(action[agent])
                    reward[agent] = torch.tensor(reward[agent])
                    next_state[agent] = torch.tensor(next_state[agent])
                    done[agent] = torch.tensor(done[agent])
                    buffer.add(current_state[agent], action[agent], reward[agent], next_state[agent], done[agent])
                    starting_point[agent] = True
                    
                if termination or truncation:
                    a = None
                else:
                    model = agent_models[id]
                    if i < 2:
                        model.coarse_tuning_settings()
                    else:
                        model.fine_tuning_settings()
                    current_state[agent] = observation
                    a = model.select_action(torch.tensor(observation), evaluate=False)
                    action[agent] = a
                    starting_point[agent] = False
                agnt = agent
                
                env.step(a)    
                buffer = agent_buffers[id]
                model = agent_models[id]
                if buffer.size >= opt.random_steps: #checks if the replay buffer has accumulated enough experiences to start training.
                    if total_steps[agnt] % opt.train_freq == 0: 
                        loss[agnt] = model.train(buffer)
                        total_training_steps[agnt]+=1
                        if opt.write:
                            wandb.log({f'training Loss for {agnt}': loss[agnt].item()})
                        model.exp_noise = schedualer.value(total_training_steps[agnt]) #e-greedy decay
                        print('episode: ',i+1,'training step: ',total_training_steps[agnt],'loss of ',agnt,': ',loss[agnt].item())
                        wandb.log({f'training step of {agnt}': total_training_steps[agnt]})

                        if total_training_steps[agnt] % opt.eval_interval == 0:
                            print("Evaluation")
                            score = evaluate_policy(rand_seed, eval_env, agent_models, agent_id, opt.write, agents)
                            print('evaluation score at the training step: ',total_training_steps[agnt],f' of the {agnt}: ', score)
                    
                        if total_training_steps[agnt] % opt.save_interval == 0:
                            print('model saving.................................')
                            if opt.transfer_train == True:
                                algo = 'dddQN_target_agent'
                                EnvName = "simple_adversary_3Good_Agents"
                            elif opt.pretrain == True:
                                algo = 'dddQN_source_agent'
                                EnvName = "simple_adversary_2Good_Agents"
                            else:
                                algo = 'dddQN_target_agent'
                                EnvName = "simple_adversary_3Good_Agents_trained_from_scratch"
                            
                            model = agent_models[id]
                            model.save(f"{algo}_{agnt}",EnvName)                    
            for agent_name in agents:
                total_game_reward[agent_name] +=  epoch_reward[agent_name]
            wandb.log({'epoch rewards': epoch_reward})     
            print('epoch rewards:', epoch_reward)           
        for agent_name in agents:
            total_game_reward[agent_name] /=epochs
        wandb.log({'avg episode rewards': total_game_reward})  
        print('avg episode rewards:', total_game_reward)   
                    
def str2bool(v):
    '''Transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True', 'true', 'TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False', 'false', 'FALSE', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.

        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)