from pettingzoo.mpe import simple_adversary_v3
import numpy as np
from ppo import Agent
import argparse
import wandb
from utils import str2bool
import torch

'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--write', type=str2bool, default=True, help='Use wandb to record the training')
parser.add_argument('--render', type=str, default="human", help='Render or Not')
parser.add_argument('--seed', type=int, default=5, help='random seed')
parser.add_argument('--save_interval', type=int, default=10, help='Model saving interval, in training steps.')
parser.add_argument('--alpha', type=float, default=0.009, help='Learning rate of the optimizer')
parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor for future rewards in the reinforcement learning problem.')
parser.add_argument('--gae_lambda', type=float, default=0.95, help='The parameter for Generalized Advantage Estimation (GAE)')
parser.add_argument('--policy_clip', type=float, default=0.2, help='Policy clipping restricts the update to a certain range (1 - policy_clip, 1 + policy_clip), preventing large policy changes.')
parser.add_argument('--batch_size', type=int, default=5, help='length of sliced trajectory')
parser.add_argument('--epoch', type=int, default=2, help='no of epochs (one epoch is one complete pass through the entire training data in the memory)')
parser.add_argument('--train_freq', type=int, default=25, help='model trainning frequency, ie. no of buffer experiences to start training')
parser.add_argument('--pretrain', type=str2bool, default=True, help='to select if pretraining on the source task is to be done')
parser.add_argument('--transfer_train', type=str2bool, default=False, help='to select if transfer learning is to be implemented or not (to be selected only after pretraining)')
parser.add_argument('--games', type=int, default=25, help='no of episodes')
parser.add_argument('--good_agents_source', type=int, default=2, help='no of good agents for the pretraining')
parser.add_argument('--good_agents_target', type=int, default=3, help='no of good agents for the target')
parser.add_argument('--best_good_agent', type=int, default=2, help='best model for the good agent')
opt = parser.parse_args()
print(opt)

def check_done_flag(termination, truncation):
    if termination or truncation:
        return True
    
if __name__ == '__main__':
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    if opt.pretrain:
        env = simple_adversary_v3.env(render_mode=opt.render, N=opt.good_agents_source, max_cycles=25, continuous_actions=False) # source task
        env_name = '1 Adv and 2 Good'
        if opt.write:
            wandb.init(project='Simple Adversary Transfer Learning', name='1 Adversary and 2 Good Agents - Pretraining (PPO)', config=vars(opt))
    else:
        env = simple_adversary_v3.env(render_mode=opt.render, N=opt.good_agents_target, max_cycles=25, continuous_actions=False) # target task
        env_name = '1 Adv and 3 Good'
        if opt.write:
            if opt.transfer_train:
                wandb.init(project='Simple Adversary Transfer Learning', name='1 Adversary and 3 Good Agents - Transfer training (all agents learned)', config=vars(opt))
            else:
                wandb.init(project='Simple Adversary Transfer Learning', name='1 Adversary and 3 Good Agents - Training from scratch', config=vars(opt))

    env.reset(seed=opt.seed)
    N = opt.train_freq
    batch_size = opt.batch_size
    n_epochs = opt.epoch
    alpha = opt.alpha
    gamma = opt.gamma
    gae_lambda = opt.gae_lambda
    policy_clip = opt.policy_clip
    agents = {}
    score = {}
    avg_score = {}
    n_steps = {}
    learn_iters ={}
    loss = {}
    score_histories = {agent_name: [] for agent_name in env.agents}

    for agent_name in env.agents:
        n_actions = env.action_space(agent_name).n
        input_dims = env.observation_space(agent_name).shape[0]
        agents[agent_name] = Agent(n_actions, input_dims, gamma, alpha, gae_lambda, policy_clip, batch_size, n_epochs, env_name, agent_name)
        avg_score[agent_name] = 0 # avg score per game
        n_steps[agent_name] = 0 # global steps
        learn_iters[agent_name] = 0
    
    n_games = opt.games
    loop_coumter = {}
    
    for i in range(n_games):
        wandb.log({f'game': i}) 
        env.reset(seed=opt.seed)
        for agent_name in env.agents:
            score[agent_name] = 0
            loop_coumter[agent_name] = 0
        
        for agent in env.agent_iter():
            loop_coumter[agent] +=1
            wandb.log({'game steps': loop_coumter}) 
            done = False
            observation, reward, termination, truncation, info = env.last()
            if check_done_flag(termination, truncation):
                action = None
            else:
#                legal_moves = np.array([observation['action_mask']])
#                game_board = np.array([observation['observation']])
                action, prob, val = agents[agent].choose_action(observation)
            
            env.step(action)
            if action == None:
                continue
            observation_, reward, termination, truncation, info = env.last()
            if check_done_flag(termination, truncation):
                done = True
            n_steps[agent] += 1
            wandb.log({'total steps': n_steps}) 

            score[agent] += reward
            wandb.log({f'score at each step for {agent}':  score[agent]})
            agents[agent].remember(observation, action, prob, val, reward, done)
            
            print('game:', i,' iteration:', loop_coumter[agent], agent,' trajectory', n_steps[agent])
            if n_steps[agent] % N == 0:
                agents[agent].learn(agent)
                learn_iters[agent] += 1
                wandb.log({f'learning steps of {agent}': learn_iters[agent]}) 
                if learn_iters[agent] % opt.save_interval == 0:
                    agents[agent].save_models()
                    print(agent, 'model saved.')
        
        print('game', i, ' terminated')
        for agent_name in env.agents:
            score_histories[agent_name].append(score[agent_name])
            avg_score[agent_name] = np.mean(score_histories[agent_name][-10:])
            wandb.log({f'avg score for {agent}':  avg_score[agent_name]})
            print('game', i, agent_name,'\'s avg score %.1f' % avg_score)
   
    env.close()
    wandb.finish()