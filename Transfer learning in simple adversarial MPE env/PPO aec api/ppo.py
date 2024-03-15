# reference: https://www.youtube.com/watch?v=hlv79rcHws0&ab_channel=MachineLearningwithPhil

import wandb
import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probabilities = []
        self.values = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size
   
    def generate_batches(self): # to create batches of experiences from the stored memory
        n_states = len(self.states) # total no of stored states
        batch_start = np.arange(0, n_states, self.batch_size)  # array of indices that represent the starting index of each batch
        indices = np.arange(n_states, dtype=np.int64) # array of indices for all stored experiences
        np.random.shuffle(indices) # shuffle the indices before batching
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probabilities),\
                np.array(self.values),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches
    
    def store_memory(self, state, action, probability, value, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probabilities.append(probability)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probabilities = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []

# The neural network that represents the policy. It outputs a probability distribution over the action space using a softmax activation.
class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, EnvName, agent_name, fc1_dims=256, fc2_dims=256, algo='PPO'):
        super(ActorNetwork, self).__init__()
        self.EnvName = EnvName
        self.algo = algo
        self.agent_name = agent_name
        self.conv_layers = nn.Sequential(                                              # input tensor shape = (batch_size, channels, height, width)
				nn.Conv1d(in_channels=1, out_channels=5, kernel_size=3, stride=1),           
				nn.ReLU(),
				nn.Conv1d(in_channels=5, out_channels=10, kernel_size=3, stride=1),            
				nn.ReLU(),
				nn.Conv1d(in_channels=10, out_channels=15, kernel_size=3, stride=1, padding=1),
				nn.ReLU(),
				nn.Conv1d(in_channels=15, out_channels=20, kernel_size=3, stride=1, padding=1),
				nn.ReLU(),
				nn.Conv1d(in_channels=20, out_channels=25, kernel_size=3, stride=1, padding=1),
				nn.ReLU(),
				nn.Conv1d(in_channels=25, out_channels=30, kernel_size=3, stride=1),   # output length = (input length + 2*padding - kernal)/stride  +1
				nn.ReLU(),
				nn.Flatten())
        self.conv_out_dim = (input_dims-6)*30
        self.fc_layers = nn.Sequential(
                nn.Linear(self.conv_out_dim, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, n_actions)
        )
        self.final_layer =  nn.Sequential(nn.Softmax(dim=-1))
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        conv_output = self.conv_layers(state)
        fc_output = self.fc_layers(conv_output)
        dist = self.final_layer(fc_output)
        dist = Categorical(dist)
        return dist, fc_output

    def save_checkpoint(self):
        save_path = "./model/{}_{}_{}_{}.pth".format(self.algo, self.EnvName, self.agent_name, "actorNet")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Create the directory if it doesn't exist
        T.save(self.state_dict(), save_path)

    def load_checkpoint(self, EnvName, agent_name):
        save_path = "./model/{}_{}_{}_{}.pth".format(self.algo, EnvName, agent_name, "actorNet")
        self.load_state_dict(T.load(save_path))

# The neural network that represents the value function. It estimates the expected return for a given state.
class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, EnvName, agent_name, fc1_dims=256, fc2_dims=256, algo='PPO'):
        super(CriticNetwork, self).__init__()
        self.EnvName = EnvName
        self.algo = algo
        self.agent_name = agent_name
        self.conv_layers = nn.Sequential(                                          # input tensor shape = (batch_size, channels, height, width)
        nn.Conv1d(in_channels=1, out_channels=5, kernel_size=3, stride=1),           
				nn.ReLU(),
				nn.Conv1d(in_channels=5, out_channels=10, kernel_size=3, stride=1),            
				nn.ReLU(),
				nn.Conv1d(in_channels=10, out_channels=15, kernel_size=3, stride=1, padding=1),
				nn.ReLU(),
				nn.Conv1d(in_channels=15, out_channels=20, kernel_size=3, stride=1, padding=1),
				nn.ReLU(),
				nn.Conv1d(in_channels=20, out_channels=25, kernel_size=3, stride=1, padding=1),
				nn.ReLU(),
				nn.Conv1d(in_channels=25, out_channels=30, kernel_size=3, stride=1),   # output length = (input length + 2*padding - kernal)/stride  +1
				nn.ReLU(),
				nn.Flatten())
        self.conv_out_dim = (input_dims-6)*30
        self.fc_layers = nn.Sequential(
                nn.Linear(self.conv_out_dim, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, 1),
                nn.Softmax(dim=-1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        conv_output = self.conv_layers(state)
        value = self.fc_layers(conv_output)
        return value

    def save_checkpoint(self):
        save_path = "./model/{}_{}_{}_{}.pth".format(self.algo, self.EnvName, self.agent_name, "criticNet")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Create the directory if it doesn't exist
        T.save(self.state_dict(), save_path)

    def load_checkpoint(self, EnvName, agent_name):
        save_path = "./model/{}_{}_{}_{}.pth".format(self.algo, EnvName, agent_name, "criticNet")
        self.load_state_dict(T.load(save_path))

class Inputnet(nn.Module):
	def __init__(self,input_obs_dim, conv_input_dim):
		super(Inputnet, self).__init__()
		self.dynamic_input_layer = nn.Sequential(
			nn.Linear(input_obs_dim, conv_input_dim)
		)
	
	def forward(self, obs):
		return self.dynamic_input_layer(obs)

class Outputnet(nn.Module):
	def __init__(self, old_action_dim, new_action_dim):
		super(Outputnet, self).__init__()
		self.dynamic_output_layer = nn.Sequential(
			nn.Linear(old_action_dim, new_action_dim)
		)
	
	def forward(self, P):
		return self.dynamic_output_layer(P)

class Combine_for_actor(nn.Module):
    def __init__(self, input_net, Actor_net, output_net, algo, EnvName, agent_name):
        super(Combine_for_actor, self).__init__()
        self.input_net = input_net
        self.Actor_net = Actor_net
        self.output_net = output_net
        self.final_layer = nn.Softmax(dim=-1)
        self.device = Actor_net.device
        self.algo = algo
        self.EnvName = EnvName
        self.agent_name = agent_name

    def forward(self, obs):
        input = self.input_net(obs)
        dist, fc_output = self.Actor_net(input)
        dist = self.output_net(fc_output)
        dist = self.final_layer(dist)
        dist = Categorical(dist)
        return dist, fc_output
	
    def get_conv_layers(self):
        return self.Actor_net.conv_layers
    
    def save_checkpoint(self):
        save_path = f"./model/{self.algo}_{self.EnvName}_{self.agent_name}_actorNet.pth"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Create the directory if it doesn't exist
        T.save(self.Actor_net.state_dict(), save_path)

class Combine_for_critic(nn.Module):
    def __init__(self, input_net, Critic_net, algo, EnvName, agent_name):
        super(Combine_for_critic, self).__init__()
        self.input_net = input_net
        self.Critic_net = Critic_net
        self.device = Critic_net.device
        self.algo = algo
        self.EnvName = EnvName
        self.agent_name = agent_name

    def forward(self, obs):
        input = self.input_net(obs)
        value = self.Critic_net(input)
        return value
	
    def get_conv_layers(self):
        return self.Critic_net.conv_layers

    def save_checkpoint(self):
        save_path = f"./model/{self.algo}_{self.EnvName}_{self.agent_name}_criticNet.pth"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Create the directory if it doesn't exist
        T.save(self.Critic_net.state_dict(), save_path)

class Agent:
    def __init__(self, n_actions, input_dims, gamma, alpha, alpha_TL, gae_lambda, policy_clip, batch_size, n_epochs, env_name, agent_name):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.alpha_TL = alpha_TL
        self.alpha = alpha
        self.actor = ActorNetwork(n_actions, input_dims, alpha, env_name, agent_name)
        self.critic = CriticNetwork(input_dims, alpha, env_name, agent_name)
        self.memory = PPOMemory(batch_size)
       
    def remember(self, state, action, probs, vals, reward, done): # interface between agent and its memory
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self, EnvName, agent_name, input_obs_dim, conv_input_dim, new_action_dim, old_action_dim, transfer_train=False):
        print('... loading models ...')
        self.actor.load_checkpoint(EnvName, agent_name)
        self.critic.load_checkpoint(EnvName, agent_name)
        if transfer_train:
            input_net = Inputnet(input_obs_dim, conv_input_dim)
            output_net = Outputnet(old_action_dim, new_action_dim)
            self.actor = Combine_for_actor(input_net, self.actor, output_net, 'PPO', EnvName, agent_name)
            self.critic = Combine_for_critic(input_net, self.critic, 'PPO', EnvName, agent_name)

    def coarse_tuning_settings(self):
        for param in self.actor.get_conv_layers().parameters():
            param.requires_grad = False
        self.actor.optimizer = optim.Adam(self.actor.parameters(), lr=self.alpha_TL)
        for param in self.critic.get_conv_layers().parameters():
            param.requires_grad = False
        self.critic.optimizer = optim.Adam(self.critic.parameters(), lr=self.alpha_TL)
	
    def fine_tuning_settings(self):
        for param in self.actor.get_conv_layers().parameters():
            param.requires_grad = True
        self.actor.optimizer = optim.Adam(self.actor.parameters(), lr=self.alpha)
        for param in self.critic.get_conv_layers().parameters():
            param.requires_grad = True
        self.critic.optimizer = optim.Adam(self.critic.parameters(), lr=self.alpha)

    def choose_action(self, observation):
        # use in case of masked actions
#        state = T.tensor(np.array([observation['observation']]), dtype=T.float32).permute(0, 3, 1, 2).to(self.actor.device) 
#        legal_moves = T.tensor(np.array([observation['action_mask']]), dtype=T.float32).to(self.actor.device)
#        masked_dist = dist.probs * legal_moves # masks out the probabilities corresponding to illegal actions.
#        if masked_dist.sum() == 0:
#            action = dist.sample() # sample any illegal action to loose the game
#        else:
#            masked_dist /= masked_dist.sum()  # Normalize to make it a valid probability distribution
#            action = Categorical(masked_dist).sample() # sample a valid action      
        state = T.tensor(observation, dtype=T.float32).to(self.actor.device) 
        state = state.view(1, 1, -1).to(self.actor.device)
        dist, fc_output = self.actor(state) #  probability distribution over actions 
          
        action = dist.sample() # sample an action
        probability = dist.log_prob(action) # log probability of the chosen action from the distribution
        value = self.critic(state) #  predicted value of the state  
        probability = T.squeeze(probability).item()  
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()
        return action, probability, value

    def learn(self,agent):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.generate_batches()
#            state_arr = np.array([obs['observation'].astype(np.float32) for obs in state_arr])
            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)
            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0 # advantage at time t
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*(1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)
            values = T.tensor(values).to(self.actor.device)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float32).to(self.actor.device)
                states = states.view(states.shape[0], 1, states.shape[1])
#                states = states.clone().detach().requires_grad_(True)
#                states = T.squeeze(states)
#                states = states.clone().detach().permute(0, 3, 1, 2)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)
                dist, fc_output = self.actor(states)
                critic_value = self.critic(states)
                critic_value = T.squeeze(critic_value)
                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                #prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()
                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()
                total_loss = actor_loss + 0.5*critic_loss              
                wandb.log({f'critic loss for {agent}': critic_loss.item()})
                wandb.log({f'actor loss for {agent}': actor_loss.item()})
                wandb.log({f'total loss for {agent}': total_loss.item()})
                print(f'total loss for {agent}:', total_loss.item())
                self.actor.optimizer.zero_grad() # sets the gradients of all model parameters to zero
                self.critic.optimizer.zero_grad()
                total_loss.backward() # Backward pass
                self.actor.optimizer.step() # Update model parameters using optimizer
                self.critic.optimizer.step()
        self.memory.clear_memory()     