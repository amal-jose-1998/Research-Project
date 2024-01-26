import copy
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import math
import time
import torch.optim as optim
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class dddQ_Net(nn.Module):
	def __init__(self, obs_dim, hidden, lr, chkpt_dir, name):
		super(dddQ_Net, self).__init__()
		if obs_dim == 8:
			i = 4
		elif obs_dim == 10:
			i = 6
		self.conv_layers = nn.Sequential(                                                  # convolutional layers
			nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1),
			nn.ReLU(),
			nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
			nn.ReLU(),
			nn.Flatten())
		self.fc_layers = nn.Sequential(	                                                   # fully connected layers
			nn.Linear(i,hidden),
			nn.ReLU()
			)                                                                      
		self.V = nn.Linear(hidden, 1)  # the value
		self.A = nn.Linear(hidden, 5)  # the advantage of actions, relative value of each action
		self.optimizer = optim.Adam(self.parameters(), lr=lr)
		self.loss = nn.MSELoss()	
		self.chkpt_dir = chkpt_dir
		self.checkpoint_file = os.path.join(self.chkpt_dir, name)

	def forward(self,obs):
		conv_output = self.conv_layers(obs)
		fc_output = self.fc_layers(conv_output)
		V = self.V(fc_output)
		A = self.A(fc_output)
		# Combining the value and advantage streams to get the Q-values
		Q = V + (A - A.mean(dim=1, keepdim=True))
		return V,A,Q
	
	# to save the state dictionary to a file in the specified path.
	def save_checkpoint(self):
		print('.......saving a checkpoint....')
		print(self.state_dict())
		torch.save(self.state_dict(), self.checkpoint_file)
	
	# to load the state dictionary from the file
	def load_checkpoint(self, file):
		print('... loading checkpoint ....')
		self.load_state_dict(torch.load(self.checkpoint_file))

	
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
		self.dw = torch.zeros((max_size, 1), dtype=torch.bool) # binary indicator that helps the learning algorithm understand when an episode or trajectory has ended.

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
	
class dddQN_Agent(object):
	def __init__(self,opt):
		self.q_net = dddQ_Net(opt.obs_dim, opt.hidden, opt.lr).to(device)
		self.q_target = copy.deepcopy(self.q_net)
		# Freeze target networks with respect to optimizers (only update via polyak averaging)
		for p in self.q_target.parameters(): p.requires_grad = False
		self.gamma = opt.gamma
		self.counter = 0
		self.batch_size = opt.batch_size
		self.action_dim = opt.action_dim
		self.DDQN = opt.DDQN
		self.target_freq = opt.target_freq
		self.hardtarget = opt.hardtarget
		self.tau = 1/opt.target_freq
		self.huber_loss = opt.huber_loss


	def select_action(self, state, evaluate):
		with torch.no_grad():
			state = state.unsqueeze(0).to(device)
			p = 0.01 if evaluate else self.exp_noise

			if np.random.rand() < p:
				a = np.random.randint(0,self.action_dim)
			else:
				a = self.q_net(state).argmax().item()

		return a


	def train(self,replay_buffer):
		s, a, r, s_prime, dw_mask = replay_buffer.sample(self.batch_size)

		'''Compute the target Q value'''
		with torch.no_grad():
			if self.DDQN:
				argmax_a = self.q_net(s_prime).argmax(dim=1).unsqueeze(-1)
				max_q_prime = self.q_target(s_prime).gather(1,argmax_a)
			else:
				max_q_prime = self.q_target(s_prime).max(1)[0].unsqueeze(1)

			'''Avoid impacts caused by reaching max episode steps'''
			target_Q = r + (1 - dw_mask) * self.gamma * max_q_prime #dw: die or win

		# Get current Q estimates
		current_q = self.q_net(s)
		current_q_a = current_q.gather(1,a)

		if self.huber_loss: q_loss = F.huber_loss(current_q_a, target_Q)
		else: q_loss = F.mse_loss(current_q_a, target_Q)

		self.q_net_optimizer.zero_grad()
		q_loss.backward()
		# for param in self.q_net.parameters(): param.grad.data.clamp_(-1, 1) #Gradient Clip
		self.q_net_optimizer.step()

		if self.hardtarget:
			# Hard update
			self.counter = (self.counter + 1) % self.target_freq
			if self.counter == 0:
				self.q_target.load_state_dict(self.q_net.state_dict())
		else:
			# Soft Update
			for param, target_param in zip(self.q_net.parameters(), self.q_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
		for p in self.q_target.parameters(): p.requires_grad = False


	def save(self,algo,EnvName,steps):
		torch.save(self.q_net.state_dict(), "./model/{}_{}_{}.pth".format(algo,EnvName,steps))

	def load(self,algo,EnvName,steps):
		self.q_net.load_state_dict(torch.load("./model/{}_{}_{}.pth".format(algo,EnvName,steps)))
		self.q_target.load_state_dict(torch.load("./model/{}_{}_{}.pth".format(algo,EnvName,steps)))


