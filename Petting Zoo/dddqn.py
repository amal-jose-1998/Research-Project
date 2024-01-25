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