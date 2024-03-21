from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from gymnasium.spaces import Discrete, MultiBinary
import numpy as np
class CustomEnvironment(AECEnv):
    metadata = {
        "name": "custom_environment_v0",
    }

    def __init__(self, system, game, render_mode = None):
        super().__init__()
        self.reward_range = (-np.inf, np.inf)
        self.system = system
        self.game = game
        self.timestep = 0
        self.resources = game.get_resources()
        self.render_mode = render_mode
        self.system_state = self.system.state
        self.NUM_ITERS = game.get_max_steps()
        self.done = False
        
        # Agents
        self.agents = game.get_players()
        self.possible_agents = self.agents.copy()
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()  

        # Spaces
        self.action_space = Discrete(len(self.system.get_actions()))                    
        self.observation_space = MultiBinary(len(self.system.events))
        self.observation = self.system.get_observations()

        # Rewards
        self.rewards = {agent : 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}                                                                                                       
        self.infos = {agent: dict() for agent in self.agents}                                                                 
        self.terminations = {agent: False for agent in self.agents}                                                                    
        self.truncations = {agent: False for agent in self.agents} 
    
    #Reset basic events to initial state received from the xml file then update intermediate events
    def reset(self, seed=None, options=None):

        self.agents = self.possible_agents.copy()
        self.timestep = 0
        self.game.reset_game()
        self.resources = self.game.initial_resources
        self.system.reset_system()
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.system_state = self.system.state
        self.infos = {agent: {} for agent in self.agents}
        self.action_space = Discrete(len(self.system.get_actions()))                    
        self.observation_space = MultiBinary(len(self.system.events))
        self.observation = self.system.get_observations()  
        if self.render_mode == "human":
            self.render()
        # print(self.agents)
        # print(self.resources)
        # print(self.rewards)
        # print(self.terminations)
        # print(self.truncations)
        # print(self.system_state)
        # print(self.infos)
        # print(self.action_space)
        # print(self.observation_space)

        return self.observation

    def step(self,act):
        to_be_deleted=[]
        red_agent, blue_agent = (self.game.players[0], self.game.players[1]) if self.game.players[0].name == "red_agent" else (self.game.players[1], self.game.players[0])
        for action in self.system.repairing_dict.keys():
            action_event = self.system.get_object(action)
            action_event.remaining_time_to_repair -=1
            if action_event.remaining_time_to_repair == 0:
                for event_name in self.system.repairing_dict[action]:
                    event = self.system.get_object(event_name)
                    event.state = 1
                to_be_deleted.append(action)
                red_agent.deactivate_action_mask(action)
        for action in to_be_deleted:
            del self.system.repairing_dict[action]
            
        # Selects agent for this step
        agent = self.agent_selection
        
        if (
            True in self.terminations.values()
            or True in self.truncations.values()
        ):
            #self._was_dead_step(action)
            return self.observation, self.rewards[agent], self.terminations, self.truncations, self.infos
        
        

        action, cost = self.game.get_action(act)
        #print(agent.name,action, cost)
        count = 0
        if cost > self.resources[agent]:
            count = self.game.apply_action(agent,'No Action')
            count = 0
        else:
            self.resources[agent] -= cost
            count = self.game.apply_action(agent, action)
        self.rewards[agent] += 1
        self.rewards[agent] += count
        #no reactive repairing instead preventive maintainance
        if agent.name == "red_agent":        
            if self.system.state == 1:
                self.rewards[agent] -= 1
            else:
                self.rewards[agent] += 10000
                # Update termination – game is "won"
                self.terminations = {agent: True for agent in self.agents}
        else:
            if self.system.state == 1:
               self.rewards[agent] += 1
            

        # Increment timestep after all agents have taken their actions
        self.timestep += 1
        self.system_state = self.system.get_system_state()
        # Update truncation - time is over.
        if self.timestep > self.NUM_ITERS:        
            self.truncations = {agent: True for agent in self.agents}
            if self.system_state == 1:
                self.rewards[blue_agent] += 1000
        self.observation = self.system.observe()
        # Infos
        self.infos = {agent : {} for agent in self.agents}

        # DEBUGGING!
        #self.infos = {"agent_red": {'time' : self.timestep, 'state' : self.system_state}, "agent_blue": {'time' : self.timestep, 'state' : self.system_state}}
        self.agent_selection = self._agent_selector.next()
        # Return observations, rewards, done, and info (customize as needed)
        # return agent.name,self.system_state, self.rewards[red_agent],self.rewards[blue_agent],self.observations, self.rewards, self.terminations, self.truncations, self.infos
        return self.observation, self.rewards[agent], self.terminations[agent], self.truncations[agent], self.infos[agent]

    def render(self):
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_space
    
    def observe(self, agent):
        return np.array(self.observation)
    