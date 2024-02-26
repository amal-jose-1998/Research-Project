import gym
import torch
import time
import argparse
import numpy as np
import wandb

def evaluate_policy(s, eval_env, agent_models, num_episodes=10):
    total_reward = {}
    for _ in range(num_episodes):
        done = False
        observations, info = eval_env.reset(seed = s+1)
        while not done:
            actions = {}
            for agent_name, model in zip(eval_env.agents, agent_models):
                state = torch.tensor(observations[agent_name], dtype=torch.float32)
                action = model.select_action(state, evaluate=True)
                actions[agent_name] = action

            next_observations, rewards, terminations, truncations, info = eval_env.step(actions)
            done = any(terminations.values()) or any(truncations.values())

            for agent_name in eval_env.agents:
                if agent_name not in total_reward:
                    total_reward[agent_name] = 0.0
                total_reward[agent_name] += rewards[agent_name]
            observations = next_observations
    observations, info = eval_env.reset()
    for agent_name in eval_env.agents:
        total_reward[agent_name] = total_reward[agent_name] / num_episodes
    return total_reward


def loop_iteration(num_games, env, eval_env, opt, agent_models, agent_buffers, good_agents):
    schedualer = LinearSchedule(schedule_timesteps=opt.anneal_frac, final_p=0.02, initial_p=opt.exp_noise) #explore noise linearly annealed from 1.0 to 0.02 within 200k steps
    loss = {}
    terminations = {}
    truncations = {}
    total_steps = 0 # total steps including the non training ones
    total_training_steps = 1 # total training step
    for i in range(num_games):
        print('episode:', i)
        total_episode_reward = {}
        actions={}
        done = False
        s, infos = env.reset(seed=opt.seed)
        for agent_name in env.agents:
            total_episode_reward[agent_name] = 0
            terminations[agent_name] = False
            truncations[agent_name] = False
        while not done:
            if any(terminations.values()) or any(truncations.values()):
                print('episode',i, 'terminated at', total_steps)
                done = 1
                wandb.log({f'total episode rewards during training': total_episode_reward}) 
            else:
                total_steps += 1
                j = 0
                for agent_name in env.agents:
                    model = agent_models[j]
                    buffer = agent_buffers[j]
                    j+=1
                    a = model.select_action(torch.tensor(s[agent_name]), evaluate=False)
                    actions[agent_name]=a

                s_prime, r, terminations, truncations, info = env.step(actions)

                j = 0
                flag = 0
                for agent_name in env.agents:
                    current_state = torch.tensor(s[agent_name])
                    next_state = torch.tensor(s_prime[agent_name])
                    reward = torch.tensor(r[agent_name])
                    action = torch.tensor(actions[agent_name])
                    if terminations[agent_name] or truncations[agent_name]:
                        done = 1
                    buffer = agent_buffers[j]
                    buffer.add(current_state, action, reward, next_state, done)
                    total_episode_reward[agent_name] += reward
                    flag = 0
                    if buffer.size >= opt.random_steps: #checks if the replay buffer has accumulated enough experiences to start training.
                        flag = 1
                        if total_steps % opt.train_freq == 0: 
                            model = agent_models[j]
                            loss[j] = model.train(buffer)
                            if opt.write:
                                wandb.log({f'training Loss for agent{j}': loss[j].item()})
                            model.exp_noise = schedualer.value(total_training_steps) #e-greedy decay
                            print('episode: ',i,'training step: ',total_training_steps,'loss of agent ',j,': ',loss[j].item())
                    j+=1

                if flag:
                    wandb.log({'training step': total_training_steps})
                    if total_training_steps % opt.eval_interval == 0:
                        score = evaluate_policy(opt.seed, eval_env, agent_models)
                        if opt.write:
                            wandb.log({'evaluation_env  avg_reward': score, 'total steps': total_steps, 'episode': i})
                            print("Evaluation")
                            print('env seed:',opt.seed+1,'evaluation score at the training step: ',total_training_steps,': ', score)
                    if total_training_steps % opt.save_interval == 0:
                        value = good_agents+1
                        if opt.transfer_train == True:
                            algo = 'dddQN_target_agent'
                            EnvName = "simple_adversary_3Good_Agents"
                        elif opt.pretrain == True:
                            algo = 'dddQN_source_agent'
                            EnvName = "simple_adversary_2Good_Agents"
                        else:
                            algo = 'dddQN_target_agent'
                            EnvName = "simple_adversary_3Good_Agents_trained_from_scratch"
                        for a in range(value):
                            model = agent_models[a]
                            model.save(f"{algo}_{a}",EnvName)                    
                    total_training_steps+=1

                s = s_prime
                
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