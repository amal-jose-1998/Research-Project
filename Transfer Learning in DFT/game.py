import random
from player import Player

class Game:
    def __init__(self,system,max_steps):
        self.system = system
        self.players=[]
        self.current_step=0
        self.max_steps = max_steps
        self.initial_resources = {}
        self.resources = {}
        self.actions = []
        self.costs = []
        self.observations = []
        self.set_actions()
        self.set_observations()
        self.set_action_costs()

    
    def reset_game(self):
        self.system = self.system.reset_system()
        self.current_step = 0
        self.resources = self.initial_resources
        for player in self.players:
            player.reset_player()
        self.set_initial_resources()
    
    def set_initial_resources(self):
        self.initial_resources = {}
        for player in self.players:
            self.initial_resources[player] = player.resources
        
        
    def set_actions(self):
        if not self.system.actions:
            self.system.set_actions()
        self.actions = self.system.get_actions()
        
    def get_actions(self):
        return self.actions
    
    def set_costs(self):
        self.costs = self.system.get_costs()

    def get_costs(self):
        return self.costs

    def set_observations(self):
        if not self.system.observations:
            self.system.set_observations()
        self.observations = self.system.get_observations()
    
    def get_observations(self):
        return self.observations
    
    def set_action_costs(self):
        self.costs = self.system.get_action_costs()

    def get_action_costs(self):
        return self.costs

    def get_system_obj(self):
        return self.system

    def get_current_step(self):
        return self.current_step
    
    def increase_step(self):
        self.current_step += 1

    def get_max_steps(self):
        return self.max_steps
    
    def get_initial_resources(self):
        return self.initial_resources
    
    def get_resources(self):
        return self.resources

    def set_player_resource(self,agent,resource):
        self.resources[agent] = resource     
    
    def get_player_resources(self, agent):
        return self.resources[agent]

    def is_game_over(self):
        if self.current_step == self.max_steps:
            return True
        else:
            return False
    
    def create_player(self, name, resources):
        player = Player(name, self.actions, resources)
        self.resources[player] = resources
        self.players.append(player)
        return player
    
    def get_players(self):
        return self.players
    
    def apply_action(self, agent, action):
        if action == "No Action":
            count = self.system.apply_action(agent.name, action)
            return count
        red_agent, blue_agent = (self.players[0], self.players[1]) if self.players[0].name == "red_agent" else (self.players[1], self.players[0])
        if agent.name=="red_agent":
            count = self.system.apply_action(red_agent.name,action)
            red_agent.activate_action_mask(action)
            blue_agent.deactivate_action_mask(action)
        else:
            count = self.system.apply_action(blue_agent.name,action)
            blue_agent.activate_action_mask(action)
        return count

    def get_action(self,action_idx):
        return self.actions[action_idx], self.costs[action_idx]
    
    def get_mask(self,player):
        return player.valid_actions_mask
    
    