import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torchsummary import summary


class Encoder(nn.Module):
	def __init__(self, obs_dim, hidden, latent_dim):
		super(Encoder, self).__init__()
		self.obs_transform = nn.Sequential(
            nn.Linear(obs_dim, 25),
            nn.ReLU(),
			nn.Linear(25, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),  # Adjust latent_dim according to your choice
        )
    
	def encode(self, obs):
		return self.obs_transform(obs)

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden, obs_dim):
        super(Decoder, self).__init__()
        self.prediction = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 25),
            nn.ReLU(),
            nn.Linear(25, obs_dim)  # Decoder with the same size as the input
        )

    def decode(self, latent_representation):
        return self.prediction(latent_representation)

class ObservationTransformer(nn.Module):
    def __init__(self, obs_dim, hidden, latent_dim):
        super(ObservationTransformer, self).__init__()
        self.obs_transform = nn.Sequential(
            nn.Linear(obs_dim, 25),
            nn.ReLU(),
			nn.Linear(25, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),  # Adjust latent_dim according to your choice
        )
		
    
    def train_obs_transform(self, obs, target_next_state, optimizer):
        latent_representation = self.obs_transform(obs)
        predicted_next_state = self.decode_latent(latent_representation)  # You need to define decode_latent
        loss = F.mse_loss(predicted_next_state, target_next_state)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()
