import numpy as np
import torch
from dddqn import dddQN_Agent, ReplayBuffer
import argparse
from utils import str2bool, loop_iteration
from pettingzoo.mpe import simple_adversary_v3
import copy
import wandb

'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--write', type=str2bool, default=True, help='Use wandb to record the training')
parser.add_argument('--render', type=str, default="human", help='Render or Not')
parser.add_argument('--seed', type=int, default=5, help='random seed')
parser.add_argument('--save_interval', type=int, default=10, help='Model saving interval, in training steps.')
parser.add_argument('--eval_interval', type=int, default=5, help='Model evaluating interval, in training steps.')
parser.add_argument('--random_steps', type=int, default=100, help=' min no of replay buffer experiences to start training')
parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='lenth of sliced trajectory')
parser.add_argument('--exp_noise', type=float, default=1.0, help='explore noise')
parser.add_argument('--buffersize', type=int, default=1e5, help='Size of the replay buffer, max 8e5')
parser.add_argument('--target_freq', type=int, default=100, help='frequency of target net updating')
parser.add_argument('--hardtarget', type=str2bool, default=False, help='True: update target net hardly(copy)')
parser.add_argument('--anneal_frac', type=int, default=2e5, help='annealing fraction of e-greedy nosise')
parser.add_argument('--train_freq', type=int, default=1, help='model trainning frequency')
parser.add_argument('--pretrain', type=str2bool, default=True, help='to select if pretraining on the source task is to be done')
parser.add_argument('--transfer_train', type=str2bool, default=False, help='to select if transfer learning is to be implemented or not (to be selected only after pretraining)')
parser.add_argument('--games', type=int, default=25, help='no of episodes')
parser.add_argument('--epoch', type=int, default=2, help='no of epochs (one epoch is one complete pass through the entire training data in the memory, even if it is by sampling)')
parser.add_argument('--good_agents_source', type=int, default=2, help='no of good agents for the pretraining')
parser.add_argument('--good_agents_target', type=int, default=3, help='no of good agents for the target')
parser.add_argument('--train_all_agents', type=str2bool, default=True, help='to select if all the agents in the target task are to be loaded with pretrained models')
parser.add_argument('--best_good_agent', type=int, default=2, help='best model for the good agent')
opt = parser.parse_args()
print(opt)

# source task
env_source = simple_adversary_v3.env(render_mode=opt.render, N=opt.good_agents_source, max_cycles=25, continuous_actions=False)
eval_env_source = simple_adversary_v3.env(render_mode=opt.render, N=opt.good_agents_source, max_cycles=25, continuous_actions=False)
source_adversary_obs_size = env_source.observation_space('adversary_0').shape[0] # 8
source_adversary_action_size = env_source.action_space('adversary_0').n # 5
source_agent_obs_size = env_source.observation_space('agent_0').shape[0]  # 10
source_agent_action_size = env_source.action_space('agent_0').n # 5
# target task
env_target= simple_adversary_v3.env(render_mode=opt.render, N=opt.good_agents_target, max_cycles=25, continuous_actions=False)
eval_env_target = simple_adversary_v3.env(render_mode=opt.render, N=opt.good_agents_target, max_cycles=25, continuous_actions=False)
target_adversary_obs_size = env_target.observation_space('adversary_0').shape[0] # 12
target_adversary_action_size = env_target.action_space('adversary_0').n # 5
target_agent_obs_size = env_target.observation_space('agent_0').shape[0]  # 14
target_agent_action_size = env_target.action_space('agent_0').n # 5

def set_observation_dimension(agent_name, pretrain = False):
    if pretrain == True:
        if agent_name=='adversary_0':
            return source_adversary_obs_size , source_adversary_action_size
        else:
            return source_agent_obs_size, source_agent_action_size
    else:
        if agent_name=='adversary_0':
            return target_adversary_obs_size, target_adversary_action_size
        else:
            return target_agent_obs_size, target_agent_action_size

def train_from_scratch(agent_models, agent_buffers, num_games, env, eval_env, n_good_agents):
    env.reset(seed=opt.seed)
    #Build model and replay buffer
    for agent_name in env.agents: 
        agent_opt = copy.deepcopy(opt)  # Create a copy of the original options for each agent
        agent_obs_dim, agent_action_dim = set_observation_dimension(agent_name, pretrain = agent_opt.pretrain)
        model = dddQN_Agent(agent_obs_dim, agent_name, agent_opt.lr, agent_opt.gamma, agent_opt.batch_size, agent_action_dim, agent_opt.target_freq, agent_opt.hardtarget, agent_opt.exp_noise, agent_opt.transfer_train)  # Create a model for each agent    
        agent_models[agent_name] = model
        buffer = ReplayBuffer(agent_obs_dim,max_size=int(agent_opt.buffersize)) # Create a replay buffer for each agent
        agent_buffers[agent_name] = buffer
    good_agents = n_good_agents
    epochs = opt.epoch
    rand_seed = opt.seed
    loop_iteration(num_games, env, eval_env, opt, agent_models, agent_buffers, good_agents, epochs,rand_seed)

if __name__ == '__main__':
    num_games = opt.games
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    agent_models = {} 
    agent_buffers = {}
    if opt.transfer_train == False:
        if opt.pretrain == True:
            if opt.write:
                wandb.init(dir="C:\\Users\\amalj\\Desktop\\wandb", project='Simple Adversary Transfer Learning', name='1 Adversary and 2 Good Agents - Pretraining (DDDQN)', config=vars(opt))
            env = env_source
            eval_env = eval_env_source
            n_good_agents = opt.good_agents_source
            train_from_scratch(agent_models, agent_buffers, num_games, env, eval_env, n_good_agents)
            env_source.close()
            eval_env_source.close()