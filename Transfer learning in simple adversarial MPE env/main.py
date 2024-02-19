import numpy as np
import torch
import gym
from dddqn import dddQN_Agent,ReplayBuffer
from transfer_learn import transfer_train
import os
from datetime import datetime
import argparse
from utils import evaluate_policy, str2bool, LinearSchedule
from pettingzoo.mpe import simple_adversary_v3
import copy
import wandb


'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--write', type=str2bool, default=True, help='Use wandb to record the training')
parser.add_argument('--render', type=str, default="human", help='Render or Not')
parser.add_argument('--seed', type=int, default=5, help='random seed')
parser.add_argument('--save_interval', type=int, default=10, help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=10, help='Model evaluating interval, in steps.')
parser.add_argument('--random_steps', type=int, default=100, help=' min no of replay buffer experiences to start training')
parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='lenth of sliced trajectory')
parser.add_argument('--exp_noise', type=float, default=1.0, help='explore noise')
parser.add_argument('--obs_dim', type=int, default=10, help='observation dimension')
parser.add_argument('--buffersize', type=int, default=1e5, help='Size of the replay buffer, max 8e5')
parser.add_argument('--target_freq', type=int, default=100, help='frequency of target net updating')
parser.add_argument('--hardtarget', type=str2bool, default=True, help='True: update target net hardly(copy)')
parser.add_argument('--action_dim', type=int, default=5, help='no of possible actions')
parser.add_argument('--anneal_frac', type=int, default=3e5, help='annealing fraction of e-greedy nosise')
parser.add_argument('--hidden', type=int, default=100, help='number of units in Fully Connected layer')
parser.add_argument('--train_freq', type=int, default=1, help='model trainning frequency')
parser.add_argument('--good_agents_pretrain', type=int, default=2, help='no of good agents for the pretraining')
parser.add_argument('--good_agents_transfer_train', type=int, default=3, help='no of good agents for the transfer training')
parser.add_argument('--transfer_train', type=str2bool, default=False, help='to select if transfer learning is to be implemented or not')
parser.add_argument('--games', type=int, default=100, help='no of episodes')
opt = parser.parse_args()
print(opt)

def main():
    num_games = opt.games
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    env_pretrain = simple_adversary_v3.parallel_env(render_mode=opt.render, N=opt.good_agents_pretrain, max_cycles=25, continuous_actions=False)
    eval_env_pretrain = simple_adversary_v3.parallel_env(render_mode=opt.render, N=opt.good_agents_pretrain, max_cycles=25, continuous_actions=False)

    env_transfer_train = simple_adversary_v3.parallel_env(render_mode=opt.render, N=opt.good_agents_transfer_train, max_cycles=25, continuous_actions=False)
    eval_env_transfer_train = simple_adversary_v3.parallel_env(render_mode=opt.render, N=opt.good_agents_transfer_train, max_cycles=25, continuous_actions=False)

    agent_models = [] # agent[0] is the adversary
    agent_buffers = []
    loss = {}
    terminations = {}
    truncations = {}

    if opt.transfer_train == False:
        if opt.write:
            wandb.init(project='Simple Adversary', name='1 Adversary and 2 Good Agents', config=vars(opt))

        #Build model and replay buffer
        for agent_id in range(opt.good_agents_pretrain+1):  
            agent_opt = copy.deepcopy(opt)  # Create a copy of the original options for each agent
            if agent_id==0:
                agent_opt.obs_dim = 8
            else:
                agent_opt.obs_dim = 10
            model = dddQN_Agent(agent_opt) # Create a model for each agent
            schedualer = LinearSchedule(schedule_timesteps=opt.anneal_frac, final_p=0.02, initial_p=opt.exp_noise) #explore noise linearly annealed from 1.0 to 0.02 within 200k steps
            model.exp_noise = opt.exp_noise
            agent_models.append(model)
            buffer = ReplayBuffer(agent_opt.obs_dim,max_size=int(opt.buffersize)) # Create a replay buffer for each agent
            agent_buffers.append(buffer)
        e = 0 # total steps including the non training ones
        total_steps = 1 # total training step
        for j in range(num_games):
            print('episode:', j)
            actions={}
            done = False
            s, infos = env_pretrain.reset()
            for agent_name in env_pretrain.agents:
                terminations[agent_name] = False
                truncations[agent_name] = False
            
            while not done:
                if any(terminations.values()) or any(truncations.values()):
                    print('episode',j, 'terminated at', e)
                    done = 1
                else:
                    e += 1
                    i = 0
                    for agent_name in env_pretrain.agents:
                        model = agent_models[i]
                        buffer = agent_buffers[i]
                        i+=1
                        a = model.select_action(torch.tensor(s[agent_name]), evaluate=False)
                        actions[agent_name]=a
                    s_prime, r, terminations, truncations, info = env_pretrain.step(actions)
                    i=0
                    flag = 0
                    for agent_name in env_pretrain.agents:
                        current_state = torch.tensor(s[agent_name])
                        next_state = torch.tensor(s_prime[agent_name])
                        reward = torch.tensor(r[agent_name])
                        action = torch.tensor(actions[agent_name])
                        if terminations[agent_name] or truncations[agent_name]:
                            done = 1
                        buffer = agent_buffers[i]
                        buffer.add(current_state, action, reward, next_state, done)
                        flag = 0
                        if buffer.size >= opt.random_steps: #checks if the replay buffer has accumulated enough experiences to start training.
                            flag = 1
                            if total_steps % opt.train_freq == 0: 
                                model = agent_models[i]
                                loss[i] = model.train(buffer)
                                if opt.write:
                                    wandb.log({f'training Loss for agent{i}': loss[i].item()})
                                #e-greedy decay
                                model.exp_noise = schedualer.value(total_steps)
                                print(model.exp_noise)
                                print('episode: ',j,'training step: ',total_steps,  'loss of agent ',i,': ',loss[i].item())
                        i+=1
                    if flag:
                        wandb.log({'training step': total_steps})
                        if total_steps % opt.eval_interval == 0:
                            score = evaluate_policy(eval_env_pretrain, agent_models)
                            if opt.write:
                                wandb.log({'reward': score, 'global step':e, 'episode': j})
                                print('seed:',opt.seed,'training steps: ',total_steps,'score:', score)

                        if total_steps % opt.save_interval == 0:
                            for a in range(opt.good_agents_pretrain+1):
                                model = agent_models[a]
                                model.save(f"dddQN_source_agent_{a}","simple_adversary_2")
                        
                        total_steps+=1

                    s = s_prime
        env_pretrain.close()
        eval_env_pretrain.close()
    
    else:
        transfer_train(opt, agent_models)

if __name__ == '__main__':
    main()