import copy
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class dddQ_Net(nn.Module):
	def __init__(self, obs_dim, hidden):
		super(dddQ_Net, self).__init__()
		if obs_dim == 8:
			i = 256
		elif obs_dim == 10:
			i = 384
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

	def forward(self,obs):
		conv_output = self.conv_layers(obs)
		fc_output = self.fc_layers(conv_output)
		V = self.V(fc_output)
		A = self.A(fc_output)
		# Combining the value and advantage streams to get the Q-values
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
		self.dw = torch.zeros((max_size, 1), dtype=torch.int8) # binary indicator that helps the learning algorithm understand when an episode or trajectory has ended.

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
	def __init__(self, opt):
		self.q_net = dddQ_Net(opt.obs_dim, opt.hidden).to(device)
		self.q_net_optimizer = optim.Adam(self.q_net.parameters(), lr=opt.lr)
		self.q_target = copy.deepcopy(self.q_net)
		self.replay_buffer = ReplayBuffer(opt.obs_dim, max_size=int(1e5))
		# Freeze target networks with respect to optimizers (only update via polyak averaging)
		for p in self.q_target.parameters(): 
			p.requires_grad = False
		self.gamma = opt.gamma             # discount factor for future rewards
		self.lr = opt.lr                   # learning rate controls the size of the steps the optimizer takes during gradient descent
		self.counter = 0                   # to keep track of the number of steps
		self.batch_size = opt.batch_size   # size of the mini-batches sampled from the replay buffer 
		self.action_dim = opt.action_dim   # the number of possible actions the agent can take in its environment
		self.target_freq = opt.target_freq # determines how often the target network is updated
		self.hardtarget = opt.hardtarget   # this flag determines whether to perform hard updates or soft updates for the target network
		self.tau = 1/opt.target_freq       # used in soft updates to control the rate at which the target network parameters are updated. It represents the interpolation factor for the weighted average between the online and target network parameters.
		self.obs_dim = opt.obs_dim
		self.exp_noise = 0

    # this function balances exploration and exploitation during decision-making. If in evaluation mode, the agent mostly exploits its knowledge, but with a small probability of exploration. 
	# if in training mode, the agent explores with a probability determined by the exploration noise parameter (self.exp_noise)
	def select_action(self, state, evaluate):
		with torch.no_grad():
			state = state.view(1, 1, -1).to(device)
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
			s_prime = s_prime.view(self.batch_size,1,s_prime.shape[1])
			argmax_a = self.q_net(s_prime)[2].argmax(dim=1).unsqueeze(-1) # action with the maximum Q-value for each sample in the next state 
			max_q_prime = self.q_target(s_prime)[2].gather(1, argmax_a) # Q-values of the chosen action in the next state from the target network 
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
    
	def save(self,algo,EnvName,steps):
		save_path = "./model/{}_{}_{}.pth".format(algo, EnvName, steps)
		os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Create the directory if it doesn't exist
		torch.save(self.q_net.state_dict(), save_path)

	def load(self,algo,EnvName,steps):
		self.q_net.load_state_dict(torch.load("./model/{}_{}_{}.pth".format(algo,EnvName,steps)))
		self.q_target.load_state_dict(torch.load("./model/{}_{}_{}.pth".format(algo,EnvName,steps)))


