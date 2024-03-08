import gym
import torch
import time
import argparse
import numpy as np
import wandb

def evaluate_policy(s, eval_env, agent_models, num_episodes=10):
    rewards = {}
    total_reward = {}
    for _ in range(num_episodes):
        eval_env.reset(seed = s+1)
        for agent in eval_env.agent_iter():
            observation, reward, termination, truncation, info = eval_env.last()
            if termination or truncation:
                a = None
                rewards[agent] = reward
            else:
                model = agent_models[agent]
                a = model.select_action(torch.tensor(observation), evaluate=True)
            eval_env.step(a)

        for agent_name in eval_env.agents:
            if agent_name not in total_reward:
                total_reward[agent_name] = 0.0
            total_reward[agent_name] += rewards[agent_name]
    for agent_name in eval_env.agents:
        total_reward[agent_name] = total_reward[agent_name] / num_episodes
    return total_reward

def check_done_flag(termination, truncation):
    if termination or truncation:
        return True

def loop_iteration(num_games, env, eval_env, opt, agent_models, agent_buffers, good_agents, epochs, rand_seed): 
    schedualer = LinearSchedule(schedule_timesteps=opt.anneal_frac, final_p=0.02, initial_p=opt.exp_noise) #exploration noise linearly annealed from 1.0 to 0.02 within 200k steps
    loss = {}
    total_steps = 0 # total steps including the non training ones
    total_training_steps = {}
    for i in range(num_games):
        print('episode:', i+1)
        for _ in range(epochs): 
            env.reset(seed=rand_seed)
            print('epoch:', _+1)
            ns_flag = {} # flag to denote whether the next state of the agent is known or not. initially it is set to 2 to denote starting
            store_flag = {} # flag to denote whether the trajectory of the agent is to be stored into the buffer or not
            next_state = {}
            action = {}
            current_state = {}
            reward = {}
            done_flag = {}
            get_next_exp ={} # flag to denote whether the next experience of the agent is to be obtained or not. initially it is set to 2 to denote starting
            
            for agent_name in env.agents:
                total_training_steps[agent_name] = 0 # total training step 
                ns_flag[agent_name] = 2
                store_flag[agent_name] = 0
                done_flag[agent_name] = 0
                get_next_exp[agent_name] = 2

            for agent in env.agent_iter():
                done_flag[agent] = 0
                observation, r, termination, truncation, info = env.last()
                
                if ns_flag[agent] == 0: 
                    next_state[agent] = torch.tensor(observation)
                    reward[agent] = torch.tensor(env.rewards[agent])
                    ns_flag[agent] = 1

                if check_done_flag(termination, truncation):
                    a = None
                    done_flag[agent] =  1
                    print('episode',i+1, f'terminated for {agent} at', total_steps)
                    wandb.log({f'total episode rewards for {agent} during training': r}) 
                else:
                    total_steps += 1
                    model = agent_models[agent]
                    a = model.select_action(torch.tensor(observation), evaluate=False)

                if a != None: 
                    store_flag[agent] = 1
                    if ns_flag[agent] == 2:
                        get_next_exp[agent] = 1
                    elif ns_flag[agent] == 1:
                        get_next_exp[agent] = 0
                else:
                    get_next_exp[agent] = 2

                if get_next_exp[agent] == 1:
                    current_state[agent] = torch.tensor(observation)
                    action[agent] = torch.tensor(a)
                    if ns_flag[agent] == 1:
                        store_flag[agent] = 1
                    ns_flag[agent] = 0

                buffer = agent_buffers[agent]
                if ns_flag[agent] == 1 and store_flag[agent] == 1:
                    get_next_exp[agent] = 1
                    ns_flag[agent] = 2
                    store_flag[agent] = 0
                    buffer.add(current_state[agent], action[agent], reward[agent], next_state[agent], done_flag[agent])
                
                env.step(a)
                if a != None: 
                    wandb.log({f'rewards for {agent} each step': env.rewards[agent]})

                if buffer.size >= opt.random_steps: #checks if the replay buffer has accumulated enough experiences to start training.
                    if total_steps % opt.train_freq == 0: 
                        loss = model.train(buffer)
                        total_training_steps[agent]+=1
                        if opt.write:
                            wandb.log({f'training Loss for {agent}': loss.item()})
                        model.exp_noise = schedualer.value(total_training_steps[agent]) #e-greedy decay
                        print('episode: ',i+1,'training step: ',total_training_steps[agent],'loss of ',agent,': ',loss.item())
                        wandb.log({f'training step of {agent}': total_training_steps[agent]})

                        if total_training_steps[agent] % opt.eval_interval == 0:
                            score = evaluate_policy(rand_seed, eval_env, agent_models)
                            if opt.write:
                                wandb.log({'avg_reward on evaluation': score, 'total steps': total_steps, 'episode': i, 'epoch': _+1})
                                print("Evaluation")
                                print('env seed:',opt.seed+1,'evaluation score at the training step: ',total_training_steps[agent],f' of the {agent}: ', score)
                        
                        if total_training_steps[agent] % opt.save_interval == 0:
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
                            
                            model = agent_models[agent]
                            model.save(f"{algo}_{agent}",EnvName)                    
                        

                    
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