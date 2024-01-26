import functools
import random
from copy import copy
import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete
from pettingzoo import ParallelEnv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class CustomEnvironment(ParallelEnv):
    # metadata holds environment constants
    metadata = {"name": "Custom_emvironment_v1",}

    # to initialise the environment attributes
    def __init__(self):
        self.escape_y = 0
        self.escape_x = 0
        self.guard_y = 0
        self.guard_x = 0
        self.prisoner_y = 0
        self.prisoner_x = 0
        self.timestep = 0
        self.possible_agents = ["prisoner", "guard"]

    # to reset the environment to a starting point
    def reset(self, seed=None, options=None):
        self.agents = copy(self.possible_agents)
        self.timestep = 0
        self.prisoner_x = 0
        self.prisoner_y = 0
        self.guard_x = 6
        self.guard_y = 6
        self.escape_x = random.randint(2, 5)
        self.escape_y = random.randint(2, 5)
        # The multiplication by 7 is to encode the grid positions uniquely.
        observations = {
            a: (
                self.prisoner_x + 7 * self.prisoner_y,
                self.guard_x + 7 * self.guard_y,
                self.escape_x + 7 * self.escape_y,
            )
            for a in self.agents
        }
        # to provide a container where additional information or metadata related to each agent can be stored
        infos = {a: {} for a in self.agents}
        return observations, infos
    
    # to go forward in the game
    def step(self,actions):
        # Execute actions
        prisoner_action = actions["prisoner"]
        guard_action = actions["guard"]
        
        if prisoner_action == 0 and self.prisoner_x > 0:
            self.prisoner_x -= 1
        elif prisoner_action == 1 and self.prisoner_x < 6:
            self.prisoner_x += 1
        elif prisoner_action == 2 and self.prisoner_y > 0:
            self.prisoner_y -= 1
        elif prisoner_action == 3 and self.prisoner_y < 6:
            self.prisoner_y += 1
        if guard_action == 0 and self.guard_x > 0:
            self.guard_x -= 1
        elif guard_action == 1 and self.guard_x < 6:
            self.guard_x += 1
        elif guard_action == 2 and self.guard_y > 0:
            self.guard_y -= 1
        elif guard_action == 3 and self.guard_y < 6:
            self.guard_y += 1        
        # Generate action masks to restrict certain movements based on the current positions of the prisoner and the guard
        prisoner_action_mask = np.ones(4, dtype=np.int8)
        if self.prisoner_x == 0:
            prisoner_action_mask[0] = 0  # Block left movement
        elif self.prisoner_x == 6:
            prisoner_action_mask[1] = 0  # Block right movement
        if self.prisoner_y == 0:
            prisoner_action_mask[2] = 0  # Block down movement
        elif self.prisoner_y == 6:
            prisoner_action_mask[3] = 0  # Block up movement
        guard_action_mask = np.ones(4, dtype=np.int8)
        if self.guard_x == 0:
            guard_action_mask[0] = 0     # Block left movement
        elif self.guard_x == 6:
            guard_action_mask[1] = 0     # Block right movement
        if self.guard_y == 0:
            guard_action_mask[2] = 0     # Block down movement
        elif self.guard_y == 6:
            guard_action_mask[3] = 0     # Block up movement
        # Action mask to prevent guard from going over escape cell
        if self.guard_x - 1 == self.escape_x:
            guard_action_mask[0] = 0     # Block left movement
        elif self.guard_x + 1 == self.escape_x:
            guard_action_mask[1] = 0     # Block right movement
        if self.guard_y - 1 == self.escape_y:
            guard_action_mask[2] = 0     # Block down movement
        elif self.guard_y + 1 == self.escape_y:
            guard_action_mask[3] = 0     # Block up movement
        # Apply action masks to the actions
        prisoner_action *= prisoner_action_mask
        guard_action *= guard_action_mask
        # Check termination conditions
        terminations = {a: False for a in self.agents}
        rewards = {a: 0 for a in self.agents}
        if self.prisoner_x == self.guard_x and self.prisoner_y == self.guard_y:
            rewards = {"prisoner": -1, "guard": 1}
            terminations = {a: True for a in self.agents}
        elif self.prisoner_x == self.escape_x and self.prisoner_y == self.escape_y:
            rewards = {"prisoner": 1, "guard": -1}
            terminations = {a: True for a in self.agents}        
        # Check truncation conditions (overwrites termination conditions)
        truncations = {a: False for a in self.agents}
        if self.timestep > 100:
            rewards = {"prisoner": 0, "guard": 0}
            truncations = {"prisoner": True, "guard": True}
        self.timestep += 1
        # Get observations
        observations = {
            a: (
                self.prisoner_x + 7 * self.prisoner_y,
                self.guard_x + 7 * self.guard_y,
                self.escape_x + 7 * self.escape_y,
            )
            for a in self.agents
        }
        # Get dummy infos 
        infos = {a: {} for a in self.agents}        
        # if any agent has reached termination conditions or if all agents have reached truncation conditions, it sets self.agents to an empty list, effectively signaling that no active agents remain in the environment
        if any(terminations.values()) or all(truncations.values()):
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    # for visually representing the current state of the environment.
    def render(self):
        grid = np.full((7, 7), " ")
        grid[self.prisoner_y, self.prisoner_x] = "P"
        grid[self.guard_y, self.guard_x] = "G"
        grid[self.escape_y, self.escape_x] = "E"
        return grid

    @functools.lru_cache(maxsize=None) # # lru_cache allows observation and action spaces to be memoized
    def observation_space(self, agent):
        return MultiDiscrete([7 * 7] * 3) # observation space consists of three discrete variables, each with a range of 7 * 7. The multiplication by 3 indicates that there are three values in the tuple for each agent.
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(4)


