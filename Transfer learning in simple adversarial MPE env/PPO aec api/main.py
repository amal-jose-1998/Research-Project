from pettingzoo.mpe import simple_adversary_v3
import numpy as np
from ppo import Agent
import argparse
import wandb
from utils import str2bool, loop_iteration
import torch

'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--render', type=str, default="human", help='Render or Not')
parser.add_argument('--seed', type=int, default=5, help='random seed')
parser.add_argument('--save_interval', type=int, default=10, help='Model saving interval, in training steps.')
parser.add_argument('--eval_interval', type=int, default=5, help='Model evaluating interval, in training steps.')
parser.add_argument('--alpha', type=float, default=0.0001, help='Learning rate of the optimizer')
parser.add_argument('--alpha_TL', type=float, default=0.001, help='Learning rate for the coarse tuning during transfer training')
parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor for future rewards in the reinforcement learning problem.')
parser.add_argument('--gae_lambda', type=float, default=0.95, help='The parameter for Generalized Advantage Estimation (GAE)')
parser.add_argument('--policy_clip', type=float, default=0.2, help='Policy clipping restricts the update to a certain range (1 - policy_clip, 1 + policy_clip), preventing large policy changes.')
parser.add_argument('--batch_size', type=int, default=10, help='length of sliced trajectory')
parser.add_argument('--epoch', type=int, default=2, help='no of epochs (one epoch is one complete pass through the entire training data in the memory)')
parser.add_argument('--train_freq', type=int, default=50, help='model trainning frequency, ie. no of buffer experiences to start training')
parser.add_argument('--pretrain', type=str2bool, default=False, help='to select if pretraining on the source task is to be done')
parser.add_argument('--transfer_train', type=str2bool, default=True, help='to select if transfer learning is to be implemented or not (to be selected only after pretraining)')
parser.add_argument('--games', type=int, default=1000, help='no of episodes')
parser.add_argument('--good_agents_source', type=int, default=2, help='no of good agents for the pretraining')
parser.add_argument('--good_agents_target', type=int, default=3, help='no of good agents for the target')
parser.add_argument('--best_good_agent', type=str, default='agent_1', help='best model for the good agent')
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

def train_from_scratch(env, eval_env):
    env.reset(seed=opt.seed)    
    N = opt.train_freq
    batch_size = opt.batch_size
    n_epochs = opt.epoch
    alpha = opt.alpha
    alpha_TL = opt.alpha_TL
    gamma = opt.gamma
    gae_lambda = opt.gae_lambda
    policy_clip = opt.policy_clip
    agent_models = []
    i = 0
    agents = []
    agent_id = {}    
    for agent_name in env.agents:
        input_dims, n_actions = set_observation_dimension(agent_name, pretrain= opt.pretrain)
        agents.append(agent_name)
        agent_id[agent_name] = i
        i+=1
        model = Agent(n_actions, input_dims, gamma, alpha, alpha_TL, gae_lambda, policy_clip, batch_size, n_epochs, env_name, agent_name)
        agent_models.append(model)
    loop_iteration(opt, agents, agent_id, agent_models, env, eval_env, N)

def set_input_layer_dimensions(agent_name): # for transfer learning
    input_dims, n_actions = set_observation_dimension(agent_name, pretrain= True)
    batch_size = opt.batch_size
    n_epochs = opt.epoch
    alpha = opt.alpha
    alpha_TL = opt.alpha_TL
    gamma = opt.gamma
    gae_lambda = opt.gae_lambda
    policy_clip = opt.policy_clip
    model = Agent(n_actions, input_dims, gamma, alpha, alpha_TL, gae_lambda, policy_clip, batch_size, n_epochs, env_name, agent_name)
    conv_input_dim = input_dims
    old_action_dim = n_actions
    obs_dim, new_action_dim = set_observation_dimension(agent_name, pretrain=False)
    input_obs_dim = obs_dim
    return input_obs_dim, conv_input_dim, new_action_dim, old_action_dim, model

def transfer_and_train(env, eval_env):
    env.reset(seed=opt.seed)    
    N = opt.train_freq
    agent_models = []
    i = 0
    agents = []
    agent_id = {}
    for agent_name in env.agents:
        agents.append(agent_name)
        agent_id[agent_name] = i
        i+=1
        input_obs_dim, conv_input_dim, new_action_dim, old_action_dim, model = set_input_layer_dimensions(agent_name)
        if agent_name != 'adversary_0': # load the model for the best trained good agent from the source task to the target task
            agent_name = opt.best_good_agent
        model.load_models("source", agent_name, input_obs_dim, conv_input_dim, new_action_dim, old_action_dim, opt.transfer_train)
        agent_models.append(model)
    loop_iteration(opt, agents, agent_id, agent_models, env, eval_env, N)
    
if __name__ == '__main__':
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    if opt.pretrain:
        env = env_source
        eval_env = eval_env_source
        env_name = 'source' 
        wandb.init(project='Simple Adversary Transfer Learning', name='1 Adversary and 2 Good Agents - Pretraining (PPO)', config=vars(opt))
        train_from_scratch(env, eval_env)      
    else:
        env = env_target
        eval_env = eval_env_target
        if opt.transfer_train:
            wandb.init(project='Simple Adversary Transfer Learning', name='1 Adversary and 3 Good Agents - Transfer training (all agents learned) PPO', config=vars(opt))
            env_name = 'target'
            transfer_and_train(env, eval_env)
        else:
            wandb.init(project='Simple Adversary Transfer Learning', name='1 Adversary and 3 Good Agents - Training from scratch (PPO)', config=vars(opt))
            env_name = 'target (from scratch)'
            train_from_scratch(env, eval_env)     
    env.close()
    eval_env.close()
    wandb.finish()