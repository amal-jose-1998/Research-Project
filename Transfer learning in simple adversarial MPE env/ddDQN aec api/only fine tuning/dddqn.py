import copy
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torchsummary import summary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
	
	def forward(self, A):
		return self.dynamic_output_layer(A)



class dddQ_Net(nn.Module):
	def __init__(self, obs_dim, agent_name):
		super(dddQ_Net, self).__init__()
		if agent_name == 'adversary_0':
			self.conv_layers = nn.Sequential(                                          # more convolutional layers for the adversary to make it strong against the agent team
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
			i = (obs_dim-6)*30

		else:
			self.conv_layers = nn.Sequential(                                          # convolutional layers for the agents
				nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, stride=1),            
				nn.ReLU(),
				nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3, stride=1),
				nn.ReLU(),
				nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=1),
				nn.ReLU(),
				nn.Conv1d(in_channels=16, out_channels=32, kernel_size=1, stride=1),
				nn.ReLU(),
				nn.Flatten())
			i = (obs_dim-6)*32
			
		self.fc_layers = nn.Sequential(	                                               # fully connected layers 
			nn.Linear(i,100),
			nn.ReLU()
			)                                                                      
		self.V = nn.Linear(100, 1)  # the value
		self.A = nn.Linear(100, 5)  # the advantage of actions, relative value of each action

	def forward(self,obs):
		conv_output = self.conv_layers(obs)
		fc_output = self.fc_layers(conv_output)
		V = self.V(fc_output)
		A = self.A(fc_output)
		# Combining the value and advantage streams to get the Q-values
		Q = V + (A - A.mean(dim=1, keepdim=True))
		return V,A,Q
	
class dddQN_Agent(object):
	def __init__(self, obs_dim, agent_name, lrate, tlrate, gamma, batch_size, action_dim, target_freq, hardtarget, exp_noise, transfer_train):
		self.q_net = dddQ_Net(obs_dim, agent_name).to(device)
		self.q_net_optimizer = optim.Adam(self.q_net.parameters(), lr=lrate)
		self.q_target = copy.deepcopy(self.q_net)
		self.replay_buffer = ReplayBuffer(obs_dim, max_size=int(1e5))
		# Freeze target networks with respect to optimizers (only update via polyak averaging)
		for p in self.q_target.parameters(): 
			p.requires_grad = False
		self.gamma = gamma                 # discount factor for future rewards
		self.lr = lrate                    # learning rate controls the size of the steps the optimizer takes during gradient descent
		self.tlr = tlrate
		self.counter = 0                   # to keep track of the number of steps
		self.batch_size = batch_size       # size of the mini-batches sampled from the replay buffer 
		self.action_dim = action_dim       # the number of possible actions the agent can take in its environment
		self.target_freq = target_freq     # determines how often the target network is updated
		self.hardtarget = hardtarget       # this flag determines whether to perform hard updates or soft updates for the target network
		self.tau = 1/target_freq           # used in soft updates to control the rate at which the target network parameters are updated. It represents the interpolation factor for the weighted average between the online and target network parameters.
		self.obs_dim = obs_dim
		self.exp_noise = exp_noise
		self.transfer_train = transfer_train
		summary(self.q_net, input_size=(1,self.obs_dim))

    # this function balances exploration and exploitation during decision-making. If in evaluation mode, the agent mostly exploits its knowledge, but with a small probability of exploration. 
	# if in training mode, the agent explores with a probability determined by the exploration noise parameter (self.exp_noise)
	def select_action(self, state, evaluate):
		with torch.no_grad():
			state = state.view(1, 1, -1).to(device) # 1d tensor to 3d tensor for the neural network (batch_size, channels, sequence_length)
			epsilon = 0.01 if evaluate else self.exp_noise
			if np.random.rand() < epsilon:
				a = np.random.randint(0,self.action_dim)
			else:
				V, A, Q = self.q_net.forward(state)
				a = torch.argmax(A).item()
		return a
	
    # for training the neural network using the Q-learning algorithm and updating the target network.
	def train(self, replay_buffer):
		s, a, r, s_prime, dw_mask = replay_buffer.sample(self.batch_size)
		# Compute the target Q value
		with torch.no_grad():
			s_prime = s_prime.view(self.batch_size, 1, s_prime.shape[1])
			argmax_a = self.q_net(s_prime)[2].argmax(dim=1).unsqueeze(-1) # action with the maximum Q-value for each sample in the next state 
			max_q_prime = self.q_target(s_prime)[2].gather(1, argmax_a) # Q-values of the chosen action in the next state, from the target network 
			target_Q = r + (1 - dw_mask) * self.gamma * max_q_prime
			
		# Get current Q estimates
		s = s.view(self.batch_size,1,s.shape[1])
		current_q = self.q_net(s)[2] # Q-values for all possible actions in the current state
		current_q_a = current_q.gather(1, a) #  selects the Q-value corresponding to the action taken in the current state.

		q_loss = F.mse_loss(current_q_a, target_Q)  
		self.q_net_optimizer.zero_grad() # Clears the gradients of the model parameters to avoid accumulation.
		q_loss.backward() # Computes the gradients of the Q-loss with respect to the model parameters using backpropagation.
		# for param in self.q_net.parameters(): param.grad.data.clamp_(-1, 1) # Gradient Clip
		self.q_net_optimizer.step() # Updates the model parameters based on the computed gradients
        
		if self.hardtarget:
			# Hard update
			self.counter = (self.counter + 1) % self.target_freq
			if self.counter == 0:
				self.q_target.load_state_dict(self.q_net.state_dict())
		else:
			# Soft Update
			for param, target_param in zip(self.q_net.parameters(), self.q_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data) #  parameters of the target network are updated as a weighted average of the online and target parameters.
		for p in self.q_target.parameters(): 
			p.requires_grad = False #  to prevent backpropagation through the target network during subsequent training steps.
       
		return q_loss 
 
	def save(self,algo,EnvName):
		save_path = "C:\\Users\\amalj\\Desktop\\ddDQN parallel api\\model\\{}_{}.pth".format(algo, EnvName)
		os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Create the directory if it doesn't exist
		torch.save(self.q_net.state_dict(), save_path)
		
	def load(self, algo, EnvName, input_obs_dim, conv_input_dim, new_action_dim, old_action_dim, lrate, transfer_train=False):
		self.q_net.load_state_dict(torch.load("C:\\Users\\amalj\\Desktop\\ddDQN parallel api\\model\\{}_{}.pth".format(algo,EnvName)))
		self.q_target.load_state_dict(torch.load("C:\\Users\\amalj\\Desktop\\ddDQN parallel api\\model\\{}_{}.pth".format(algo,EnvName)))
		if transfer_train:
			self.obs_dim = input_obs_dim
			self.action_dim = new_action_dim
			input_net = Inputnet(self.obs_dim, conv_input_dim)
			output_net = Outputnet(old_action_dim, self.action_dim)
			self.q_net = Combine(input_net, self.q_net, output_net)
			self.q_net_optimizer = optim.Adam(self.q_net.parameters(), lr=lrate)
			print("target task network")
			summary(self.q_net, input_size=(1,self.obs_dim))
			self.q_target = Combine(input_net, self.q_target, output_net) 
	
	def coarse_tuning_settings(self):
		for param in self.q_net.conv_layers.parameters():
			param.requires_grad = False
		self.q_net_optimizer = optim.Adam(self.q_net.parameters(), lr=self.tlr)
	
	def fine_tuning_settings(self):
		for param in self.q_net.conv_layers.parameters():
			param.requires_grad = True
		self.q_net_optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)



class Combine(nn.Module):
	def __init__(self,input_net, q_net, output_net):
		super(Combine, self).__init__()
		self.input_net = input_net
		self.q_net = q_net
		self.output_net = output_net
		# Set requires_grad to False for convolutional layers if it is to be frozen and not tuned in the target task
		for param in self.q_net.conv_layers.parameters():
			param.requires_grad = True

	def forward(self, obs):
		input = self.input_net(obs)
		V,A,_ = self.q_net(input)
		A = self.output_net(A)
		Q = V + (A - A.mean(dim=1, keepdim=True))
		return V,A,Q

class ReplayBuffer():
	def __init__(self, obs_dim, max_size=int(1e5)):
		self.device = device # device on which the PyTorch tensors will be stored (CPU or GPU).
		self.max_size = max_size
		self.ptr = 0 # index pointing to the location where the next memory will be stored
		self.size = 0 # current number of experiences stored in the replay buffer

		self.state = torch.zeros((max_size, obs_dim))
		self.action = torch.zeros((max_size, 1), dtype=torch.int64)
		self.reward = torch.zeros((max_size, 1))
		self.next_state = torch.zeros((max_size, obs_dim))
		self.dw = torch.zeros((max_size, 1), dtype=torch.int8) # binary indicator that helps the learning algorithm understand when an episode has ended.

	def add(self, state, action, reward, next_state, dw):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.reward[self.ptr] = reward
		self.next_state[self.ptr] = next_state
		self.dw[self.ptr] = dw  # 0,0,0，...，1

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		ind = np.random.choice((self.size-1), batch_size, replace=False)  # Time consuming, but no duplication
		# ind = np.random.randint(0, (self.size-1), batch_size)  # Time effcient, might duplicates
		return (
			self.state[ind].to(self.device),
			self.action[ind].to(self.device),
			self.reward[ind].to(self.device),
			self.next_state[ind].to(self.device),
			self.dw[ind].to(self.device)
		)	


