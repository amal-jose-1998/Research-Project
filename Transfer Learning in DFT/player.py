class Player:
    def __init__(self, name, actions, resources):
        self.name = name
        self.resources = resources
        self.valid_actions = [action for action in actions]
        self.valid_actions_mask = []
        self.score = 0
        self.took_invalid_action = False
        self.reset_masks()
    
    def increase_score(self):
        self.score = self.score + 1
    
    def get_valid_actions(self): 
        return self.valid_actions
    
    def get_num_valid_actions(self): 
        return len(self.valid_actions)
    
    def get_score(self):
        return self.score
    
    def reset_player(self):
        self.score = 0
        self.reset_masks()
        
    
    def activate_action_mask(self, action):
        idx = self.valid_actions.index(action)
        self.valid_actions_mask[idx] = 1

    def deactivate_action_mask(self, action):
        idx = self.valid_actions.index(action)
        self.valid_actions_mask[idx] = 0     
    
    def reset_masks(self):
        if self.name == "red_agent":
            self.valid_actions_mask = [1]*len(self.valid_actions)
        else:
            self.valid_actions_mask = [1]
            self.valid_actions_mask.extend([0]*(len(self.valid_actions) - 1))
    
    



        