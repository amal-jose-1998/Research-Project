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
        for agent in eval_env.agents:
            rewards[agent.name] = 0
        for agent in eval_env.agent_iter():
            if all([eval_env.terminations[a] for a in eval_env.terminations]) or all([eval_env.truncations[a] for a in eval_env.truncations]):
                break
            id = agent_id[agent.name]
            observation, cr, termination, truncation, info = eval_env.last()
            if termination or truncation:
                a = None
            else:
                model = agent_models[id]
                a = model.select_action(torch.tensor(observation), torch.tensor(agent.valid_actions_mask), evaluate=True)
            agnt = agent
            observation_, r, termination, truncation, info = eval_env.step(a)
            rewards[agnt.name] += r

        for agent in agents:
            if agent not in total_reward:
                total_reward[agent] = 0.0
            total_reward[agent] += rewards[agent]
    
    for agent in agents:
        total_reward[agent] = total_reward[agent] / num_episodes
        if write:
            wandb.log({f'avg_reward on evaluation for {agent}': total_reward[agent]})
            print(f'avg_reward on evaluation for {agent}: ', total_reward[agent])
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

def reset_coarse_tuner_counters(agents, coarse_tuner_counter):
    for s in agents:
        coarse_tuner_counter[s] = 0
        
def loop_iteration(num_games, env, eval_env, opt, agent_models, agent_buffers, agents, agent_id, epochs, rand_seed): 
    schedualer = LinearSchedule(schedule_timesteps=opt.anneal_frac, final_p=0.02, initial_p=opt.exp_noise) #exploration noise linearly annealed from 1.0 to 0.02 within 200k steps
    loss = {}
    total_training_steps = {}
    total_steps = {} 
    total_game_reward = {}
    epoch_reward = {}
    coarse_tuner_counter = {}

    # coarse tuning for the initial steps during transfer learning
    if opt.transfer_train == True:
        reset_coarse_tuner_counters(agents, coarse_tuner_counter)

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
            next_state = {}
            current_state = {}
            action = {}
            reward = {}
            done = {}
        
            for agent in env.agent_iter():
                if all([env.terminations[a] for a in env.terminations]) or all([env.truncations[a] for a in env.truncations]):
                    print('epoch:', _+1, ' of the Game ',i+1, 'terminated at', total_steps[agent.name])
                    break
                total_steps[agent.name]+=1
                wandb.log({f'total_steps {agent.name}': total_steps[agent.name]})
                observation, cr, termination, truncation, info = env.last()
                id = agent_id[agent.name]
                current_state[agent.name] = observation
                if termination or truncation:
                    a = None
                else:
                    buffer = agent_buffers[id]
                    model = agent_models[id]

                    if opt.transfer_train == True:
                        if buffer.size >= opt.random_steps:
                            if coarse_tuner_counter[agent.name] < 50:
                                model.coarse_tuning_settings()
                                coarse_tuner_counter[agent.name]+=1
                                print("coarse tuning for ", agent.name)
                            else:
                                model.fine_tuning_settings()

                    a = model.select_action(torch.tensor(observation), torch.tensor(agent.valid_actions_mask), evaluate=False)
                    action[agent.name] = a
                
                agnt = agent
                observation_, r, termination, truncation, info = env.step(a) 

                if termination or truncation:
                    done[agnt.name] = 1
                else:
                    done[agnt.name] = 0    
                next_state[agnt.name] = observation_
                reward[agnt.name] = r
                wandb.log({f'step reward for {agnt.name}': r})  
                print(f'step reward for {agnt.name} at step {total_steps[agnt.name]}:', r)
                epoch_reward[agnt.name]+=r
                    
                buffer = agent_buffers[id]
                model = agent_models[id]
                current_state[agnt.name] = torch.tensor(current_state[agnt.name])
                action[agnt.name] = torch.tensor(action[agnt.name])
                reward[agnt.name] = torch.tensor(reward[agnt.name])
                next_state[agnt.name] = torch.tensor(next_state[agnt.name])
                done[agnt.name] = torch.tensor(done[agnt.name])
                buffer.add(current_state[agnt.name], action[agnt.name], reward[agnt.name], next_state[agnt.name], done[agnt.name])
                
                if buffer.size >= opt.random_steps: #checks if the replay buffer has accumulated enough experiences to start training.
                    if total_steps[agnt.name] % opt.train_freq == 0: 
                        loss[agnt.name] = model.train(buffer, opt.huber_loss)
                        total_training_steps[agnt.name]+=1
                        if opt.write:
                            wandb.log({f'training Loss for {agnt.name}': loss[agnt.name].item()})
                        model.exp_noise = schedualer.value(total_training_steps[agnt.name]) #e-greedy decay
                        print('episode: ',i+1,'training step: ',total_training_steps[agnt.name],'loss of ',agnt.name,': ',loss[agnt.name].item())
                        wandb.log({f'training step of {agnt.name}': total_training_steps[agnt.name]})

                        if total_training_steps[agnt.name] % opt.eval_interval == 0:
                            print("Evaluation")
                            score = evaluate_policy(rand_seed, eval_env, agent_models, agent_id, opt.write, agents)
                            print('evaluation score at the training step: ',total_training_steps[agnt.name],f' of the {agnt.name}: ', score)
                    
                        if total_training_steps[agnt.name] % opt.save_interval == 0:
                            print('model saving.................................')
                            if opt.transfer_train == True:
                                algo = 'target'
                                EnvName = "Transfer_Learned"
                            elif opt.pretrain == True:
                                algo = 'source'
                                EnvName = "Pretrained"
                            else:
                                algo = 'target'
                                EnvName = "trained_from_scratch"
                            
                            model = agent_models[id]
                            model.save(f"{algo}_{agnt.name}",EnvName)                    
            for agent in agents:
                total_game_reward[agent] +=  epoch_reward[agent]
            wandb.log({'epoch rewards': epoch_reward})     
            print('epoch rewards:', epoch_reward)           
        for agent in agents:
            total_game_reward[agent] /=epochs
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