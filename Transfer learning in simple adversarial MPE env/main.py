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
parser.add_argument('--obs_dim', type=int, default=10, help='observation dimension')
parser.add_argument('--buffersize', type=int, default=1e5, help='Size of the replay buffer, max 8e5')
parser.add_argument('--target_freq', type=int, default=100, help='frequency of target net updating')
parser.add_argument('--hardtarget', type=str2bool, default=False, help='True: update target net hardly(copy)')
parser.add_argument('--action_dim', type=int, default=5, help='no of possible actions')
parser.add_argument('--anneal_frac', type=int, default=2e5, help='annealing fraction of e-greedy nosise')
parser.add_argument('--hidden', type=int, default=100, help='number of units in Fully Connected layer')
parser.add_argument('--train_freq', type=int, default=1, help='model trainning frequency')
parser.add_argument('--good_agents_pretrain', type=int, default=2, help='no of good agents for the pretraining')
parser.add_argument('--good_agents_target', type=int, default=3, help='no of good agents for the target')
parser.add_argument('--pretrain', type=str2bool, default=False, help='to select if pretraining on the source task is to be done')
parser.add_argument('--transfer_train', type=str2bool, default=True, help='to select if transfer learning is to be implemented or not (to be selected only after pretraining)')
parser.add_argument('--games', type=int, default=60, help='no of episodes')
parser.add_argument('--train_all_agents', type=str2bool, default=True, help='to select if all the agents in the target task are to be loaded with pretrained models')
parser.add_argument('--best_good_agent', type=int, default=2, help='best model for the good agent')
opt = parser.parse_args()
print(opt)

def set_observation_dimension(agent_id, transfer_train=False, pretrain = False):
    if transfer_train == False:
        if pretrain == True:
            if agent_id==0:
                return 8
            else:
                return 10
        else:
            if agent_id==0:
                return 12
            else:
                return 14
    else:
        if agent_id==0:
            return 12
        else:
            return 14

def set_input_layer_dimensions(agent_id, agent_opt):
    agent_opt.obs_dim = set_observation_dimension(agent_id, pretrain= True)
    model = dddQN_Agent(agent_opt, agent_id) # Create a model for each agent
    conv_input_dim = agent_opt.obs_dim
    obs_dim = set_observation_dimension(agent_id, transfer_train=True)
    input_obs_dim = obs_dim
    return input_obs_dim,conv_input_dim, model

def main():
    num_games = opt.games
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    env_pretrain = simple_adversary_v3.parallel_env(render_mode=opt.render, N=opt.good_agents_pretrain, max_cycles=25, continuous_actions=False)
    eval_env_pretrain = simple_adversary_v3.parallel_env(render_mode=opt.render, N=opt.good_agents_pretrain, max_cycles=25, continuous_actions=False)
    env_transfer_train = simple_adversary_v3.parallel_env(render_mode=opt.render, N=opt.good_agents_target, max_cycles=25, continuous_actions=False)
    eval_env_transfer_train = simple_adversary_v3.parallel_env(render_mode=opt.render, N=opt.good_agents_target, max_cycles=25, continuous_actions=False)
    env_train_from_scratch = simple_adversary_v3.parallel_env(render_mode=opt.render, N=opt.good_agents_target, max_cycles=25, continuous_actions=False)
    eval_env_train_from_scratch= simple_adversary_v3.parallel_env(render_mode=opt.render, N=opt.good_agents_target, max_cycles=25, continuous_actions=False)


    agent_models = [] # agent[0] is the adversary
    agent_buffers = []
    
    if opt.transfer_train == False:
        if opt.pretrain == True:
            if opt.write:
                wandb.init(project='Simple Adversary Transfer Learning', name='1 Adversary and 2 Good Agents - Pretraining', config=vars(opt))
            print(env_pretrain.observation_space)
            #Build model and replay buffer
            for agent_id in range(opt.good_agents_pretrain+1):  
                agent_opt = copy.deepcopy(opt)  # Create a copy of the original options for each agent
                agent_opt.obs_dim = set_observation_dimension(agent_id, pretrain = True)
                model = dddQN_Agent(agent_opt, agent_id) # Create a model for each agent    
                agent_models.append(model)
                buffer = ReplayBuffer(agent_opt.obs_dim,max_size=int(opt.buffersize)) # Create a replay buffer for each agent
                agent_buffers.append(buffer)
            
            good_agents = opt.good_agents_pretrain
            loop_iteration(num_games, env_pretrain, eval_env_pretrain, opt, agent_models, agent_buffers, good_agents)
            env_pretrain.close()
            eval_env_pretrain.close()
       
        else:
            if opt.write:
                wandb.init(project='Simple Adversary Transfer Learning', name='1 Adversary and 3 Good Agents - Training from scratch', config=vars(opt))
            print(env_train_from_scratch.observation_space)
            #Build model and replay buffer
            for agent_id in range(opt.good_agents_target+1):  
                agent_opt = copy.deepcopy(opt)  # Create a copy of the original options for each agent
                agent_opt.obs_dim = set_observation_dimension(agent_id)
                model = dddQN_Agent(agent_opt, agent_id) # Create a model for each agent    
                agent_models.append(model)
                buffer = ReplayBuffer(agent_opt.obs_dim,max_size=int(opt.buffersize)) # Create a replay buffer for each agent
                agent_buffers.append(buffer)

            good_agents = opt.good_agents_target
            loop_iteration(num_games, env_train_from_scratch, eval_env_train_from_scratch, opt, agent_models, agent_buffers, good_agents)
            env_train_from_scratch.close()
            eval_env_train_from_scratch.close()

    
    else:
        if opt.write:
            if opt.train_all_agents == False:
                wandb.init(project='Simple Adversary Transfer Learning', name='1 Adversary and 3 Good Agents - Transfer training (only corresponding agents learned)', config=vars(opt))
            else:
                wandb.init(project='Simple Adversary Transfer Learning', name='1 Adversary and 3 Good Agents - Transfer training (all agents learned)', config=vars(opt))
        print(env_transfer_train.observation_space)
        for agent_id in range(opt.good_agents_target+1):  
            agent_opt = copy.deepcopy(opt)  # Create a copy of the original options for each agent
            if opt.train_all_agents == False:
                if agent_id < opt.good_agents_pretrain: 
                    input_obs_dim, conv_input_dim, model = set_input_layer_dimensions(agent_id, agent_opt)
                    model.load(f"dddQN_source_agent_{agent_id}","simple_adversary_2Good_Agents", input_obs_dim, conv_input_dim, opt.transfer_train) # Load pretrained models for the first two good agents and the adversary  
                else:
                    agent_opt.obs_dim = set_observation_dimension(agent_id, transfer_train=True)
                    model = dddQN_Agent(agent_opt, agent_id) # Create a model for each agent 
            else:
                input_obs_dim, conv_input_dim, model = set_input_layer_dimensions(agent_id, agent_opt)
                if agent_id!=0:
                    agent_id=opt.best_good_agent
                print(agent_id, input_obs_dim, conv_input_dim)
                model.load(f"dddQN_source_agent_{agent_id}","simple_adversary_2Good_Agents", input_obs_dim, conv_input_dim, opt.transfer_train)
           
            agent_models.append(model)
            
            agent_opt.obs_dim = set_observation_dimension(agent_id, transfer_train=True)
            buffer = ReplayBuffer(agent_opt.obs_dim,max_size=int(opt.buffersize)) # Create a replay buffer for each agent
            agent_buffers.append(buffer)             
        
        good_agents = opt.good_agents_target
        loop_iteration(num_games, env_transfer_train, eval_env_transfer_train, opt, agent_models, agent_buffers, good_agents)
        env_transfer_train.close()
        eval_env_transfer_train.close()

if __name__ == '__main__':
    main()
