import copy
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import math
import time
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class dddQ_Net(nn.Module):
	def __init__(self, obs_dim, hidden = 128, lr=0.001):
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
			nn.ReLU
			)                                                                      
		self.V = nn.Linear(hidden, 1)  # the value
		self.A = nn.Linear(hidden, 5)  # the advantage of actions, relative value of each action
		self.optimizer = optim.Adam(self.parameters(), lr=lr)
		self.loss = nn.MSELoss()	

	def forward(self,obs):
		conv_output = self.conv_layers(obs)
		fc_output = self.fc_layers(conv_output)
		V = self.V(fc_output)
		A = self.A(fc_output)
		# Combining the value and advantage streams to get the Q-values
		Q = V + (A - A.mean(dim=1, keepdim=True))
		return Q
	
class ReplayBuffer():
	def __init__(self, max_size=int(1e5)):
		self.device = device
		self.max_size = max_size
		self.ptr = 0 #index of the last stored memmory
		self.size = 0

		self.state = torch.zeros((max_size, 4, 84, 84))
		self.action = torch.zeros((max_size, 1), dtype=torch.int64)
		self.reward = torch.zeros((max_size, 1))
		self.next_state = torch.zeros((max_size, 4, 84, 84))
		self.dw = torch.zeros((max_size, 1), dtype=torch.int8)

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
		return self.state[ind].to(self.device),self.action[ind].to(self.device),self.reward[ind].to(self.device),\
			   self.next_state[ind].to(self.device),self.dw[ind].to(self.device)