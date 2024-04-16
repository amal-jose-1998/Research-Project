import numpy as np
import torch
from dddqn import dddQN_Agent, ReplayBuffer
import argparse
from utils import str2bool, loop_iteration
from env_creator import env_creator
import copy
import wandb

'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--write', type=str2bool, default=True, help='Use wandb to record the training')
parser.add_argument('--seed', type=int, default=5, help='random seed')

parser.add_argument('--games', type=int, default=15, help='no of episodes')
parser.add_argument('--epoch', type=int, default=2, help='no of epochs (one epoch is one complete pass through the entire training data in the memory, even if it is by sampling)')
parser.add_argument('--pretrain', type=str2bool, default=False, help='to select if pretraining on the source task is to be done')
parser.add_argument('--transfer_train', type=str2bool, default=False, help='to select if transfer learning is to be implemented or not (to be selected only after pretraining)')
parser.add_argument('--source_dim', type=str, default='high', help='Use low/high dimensional model as source for pretraining')

parser.add_argument('--save_interval', type=int, default=10, help='Model saving interval, in training steps.')
parser.add_argument('--eval_interval', type=int, default=5, help='Model evaluating interval, in training steps.')
parser.add_argument('--random_steps', type=int, default=100, help=' min no of replay buffer experiences to start training')
parser.add_argument('--target_freq', type=int, default=100, help='frequency of target net updating')
parser.add_argument('--train_freq', type=int, default=1, help='model trainning frequency')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate - ordinary')
parser.add_argument('--tlr', type=float, default=0.001, help='Learning rate - coarse tuning during transfer learning')
parser.add_argument('--batch_size', type=int, default=32, help='lenth of sliced trajectory')
parser.add_argument('--exp_noise', type=float, default=1.0, help='explore noise')
parser.add_argument('--buffersize', type=int, default=1e5, help='Size of the replay buffer, max 8e5')
parser.add_argument('--hardtarget', type=str2bool, default=False, help='True: update target net hardly(copy)')
parser.add_argument('--anneal_frac', type=int, default=2e5, help='annealing fraction of e-greedy nosise')
parser.add_argument('--huber_loss', type=str2bool, default=True, help='True: use huber_loss; False:use mse_loss')


opt = parser.parse_args()
print(opt)
# to select the appropriate source and target tasks depending on the experiment being conducted
if opt.source_dim == 'low':
    file_path_source = 'model_low_dimension.xml'
    file_path_target = 'model_high_dimension.xml'
    pro = "low source"
else:
    file_path_source = 'model_high_dimension.xml'
    file_path_target = 'model_low_dimension.xml'
    pro = "high source"

# source task - env creation 
env_source = env_creator.create_env(file_path_source)
eval_env_source = env_creator.create_env(file_path_source)
source_obs_size = env_source.observation_space.shape[0] 
source_action_size = env_source.action_space.n 

# target task - env creation
env_target= env_creator.create_env(file_path_target)
eval_env_target = env_creator.create_env(file_path_target)
target_obs_size = env_target.observation_space.shape[0] 
target_action_size = env_target.action_space.n 

# returns the dimensions of the observation and action space according to the task (ie. source task, if pretrain is true; target task, if pretrain is false)
def set_observation_dimension(pretrain = False):
    if pretrain == True:
        return source_obs_size , source_action_size
    else:
        return target_obs_size, target_action_size
    
# used to accomodate the changed dimensions of the observation and action space, in the target task
def set_input_layer_dimensions(agent_name, agent_opt): # used only for transfer learning
    agent_obs_dim, agent_action_dim = set_observation_dimension(pretrain= True)
    model = dddQN_Agent(agent_obs_dim, agent_name, agent_opt.lr, agent_opt.tlr, agent_opt.gamma, agent_opt.batch_size, agent_action_dim, agent_opt.target_freq, agent_opt.hardtarget, agent_opt.exp_noise, agent_opt.transfer_train)
    conv_input_dim = agent_obs_dim                                      # observation dimension of the source task
    old_action_dim = agent_action_dim                                   # action space dimension for the source task
    obs_dim, action_dim = set_observation_dimension(pretrain=False)
    input_obs_dim = obs_dim                                             # observation dimension of the target task
    new_action_dim = action_dim                                         # action space dimension for the target task
    return input_obs_dim, conv_input_dim, new_action_dim, old_action_dim, model

# used to train the agents on the chosen task from scratch without any transfer learning
def train_from_scratch(agent_models, agent_buffers, num_games, env, eval_env):
    env.reset(seed=opt.seed)
    i = 0
    agents = []
    agent_id = {}
    #Build model and replay buffer
    for agent in env.agents: 
        agents.append(agent.name)
        agent_id[agent.name] = i
        i+=1
        agent_opt = copy.deepcopy(opt)  # Create a copy of the original options for each agent
        agent_obs_dim, agent_action_dim = set_observation_dimension(agent_opt.pretrain)
        model = dddQN_Agent(agent_obs_dim, agent.name, agent_opt.lr, agent_opt.tlr, agent_opt.gamma, agent_opt.batch_size, agent_action_dim, agent_opt.target_freq, agent_opt.hardtarget, agent_opt.exp_noise, agent_opt.transfer_train)  # Create a model for each agent    
        agent_models.append(model)
        buffer = ReplayBuffer(agent_obs_dim,max_size=int(agent_opt.buffersize)) # Create a replay buffer for each agent
        agent_buffers.append(buffer)
    epochs = opt.epoch
    rand_seed = opt.seed
    loop_iteration(num_games, env, eval_env, opt, agent_models, agent_buffers, agents, agent_id, epochs, rand_seed)

# for transfer learning
def transfer_and_train(agent_models, agent_buffers, num_games, env, eval_env):
    env.reset(seed=opt.seed)
    i = 0
    agents = []
    agent_id = {}
    for agent in env.agents: 
        agents.append(agent.name)
        agent_id[agent.name] = i
        i+=1
        agent_opt = copy.deepcopy(opt)  
        # load the trained model, add extra input and output layers to match the new observation and action space dimensions (of the target task)
        input_obs_dim, conv_input_dim, new_action_dim, old_action_dim, model = set_input_layer_dimensions(agent.name, agent_opt)
        model.load(f"source_{agent.name}","Pretrained", input_obs_dim, conv_input_dim, new_action_dim, old_action_dim, agent_opt.lr, agent_opt.transfer_train)
        agent_models.append(model)
        agent_obs_dim, agent_action_dim = set_observation_dimension(pretrain=False)
        buffer = ReplayBuffer(agent_obs_dim,max_size=int(opt.buffersize)) # Create a replay buffer for each agent
        agent_buffers.append(buffer)     
    epochs = opt.epoch
    rand_seed = opt.seed
    loop_iteration(num_games, env, eval_env, opt, agent_models, agent_buffers, agents, agent_id, epochs, rand_seed)

# main function
if __name__ == '__main__':
    num_games = opt.games
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    agent_models = []
    agent_buffers = []
    if opt.transfer_train == False:
        if opt.pretrain == True:
            if opt.write:
                wandb.init(dir="C:\\Users\\amalj\\Desktop\\wandb", project=f'Transfer Learning for the Adversarial Game in DFT ({pro})', name='Pretraining', config=vars(opt))
            env = env_source
            eval_env = eval_env_source
            train_from_scratch(agent_models, agent_buffers, num_games, env, eval_env)
            env_source.close()
            eval_env_source.close()
        else:
            if opt.write:
                wandb.init(dir="C:\\Users\\amalj\\Desktop\\wandb",project=f'Transfer Learning for the Adversarial Game in DFT ({pro})', name='Target Task - Training from scratch', config=vars(opt))
            env = env_target
            eval_env = eval_env_target
            train_from_scratch(agent_models, agent_buffers, num_games, env, eval_env)
            env_target.close()
            eval_env_target.close()
    else:
        if opt.write:
            wandb.init(dir="C:\\Users\\amalj\\Desktop\\wandb",project=f'Transfer Learning for the Adversarial Game in DFT ({pro})', name='Target Task - Transfer training', config=vars(opt))
        env = env_target
        eval_env = eval_env_target
        transfer_and_train(agent_models, agent_buffers, num_games, env, eval_env)
        env_target.close()
        eval_env_target.close()