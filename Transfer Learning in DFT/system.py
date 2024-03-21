from element import Event, No_Action
class System:
    def __init__(self):
        self.events = {}
        self.precedences = []
        self.state = 0
        self.actions = []
        self.costs = {}
        self.observations = []
        self.repairing_dict = {}

    def add_event(self, event):
        self.events[event.name] = event

    def add_precedence(self, precedence):
        self.precedences.append(precedence)
        source=self.get_object(precedence.source)
        target=self.get_object(precedence.target)
        target.input.append(source)
        source.output.append(target)
        if precedence.precedence_type=='CSP':
            competitor=self.get_object(precedence.competitor)
            target.competitor=competitor
            target.spare=source
            target.using_spare=1
    
    def get_top_event(self):
        for event in self.events.values():
            if event.event_type=="TOP":
                return event
    
    def reset_system(self):
        for event in self.events.values():
            if event.event_type=='BASIC':
                event.state=event.initial_state
        Event.update_event(self.top_event)
        self.state = self.top_event.state
        self.repairing_dict = {}
        return self
        

    def initialize_system(self):
        self.top_event=self.get_top_event()
        self.reset_system()
        Event.update_event(self.top_event)
        
        self.set_actions()
        self.set_observations()

    def set_actions(self):
        no_action = No_Action('No Action')
        self.add_event(no_action)
        actions = [no_action.name]
        basic_events = list(self.get_basicEvents())
        for event in basic_events:
            actions.append(event.name)
        self.actions = actions
    
    def get_actions(self):
        return self.actions
    
    def set_observations(self):
        event_list=list(self.events.keys())
        event_list.sort()
        observations=[]
        for event in event_list:
            observations.append(self.events[event].state)
        self.observations = observations
        
    def set_costs(self):
        costs = [0]
        basic_event = list(self.get_basicEvents())
        for event in basic_event.sort():
            costs.append(event.failure_cost)
        self.costs = costs
    
    def get_costs(self):
        return self.costs
    
    def get_observations(self):
        return self.observations

    def get_object(self,object):
        return self.events.get(object, None)
    
    def num_actions(self):
        if not self.actions:
            return len(self.set_actions())
        else:
            return len(self.actions)
    
    def num_observations(self):
        if not self.observations:
            return len(self.set_observations())
        else:
            return len(self.observations)

    def apply_action(self, agent, action):
        #print(action)
        event = self.get_object(action)
        if agent == 'red_agent':
            count = event.red_action()
        elif agent == 'blue_agent':
            count,visited = event.blue_action()
            if not action == "No Action":
                self.repairing_dict[action] = visited
                #print(visited)
        return count

    def get_events(self):
        return self.events
    
    def get_basicEvents(self):
        return [event for event in self.events.values() if event.event_type == "BASIC"]
                
    def get_action_costs(self):
        costs = [0]
        for event in self.get_basicEvents():
            costs.append(int(event.failure_cost))
        return costs
    
    def get_system_state(self):
        self.state = self.get_top_event().state
        return self.state
    
    def observe(self):
        self.set_observations()
        return self.get_observations()