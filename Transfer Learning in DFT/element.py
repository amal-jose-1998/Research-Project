class Event:
    count = 0
    red_visited = []
    blue_visited = []
    def __init__(self, name, event_type, state=1):
        self.name = name
        self.event_type = event_type
        self.initial_state=int(state)
        self.state = int(state)
        self.input = []
        self.old_state=int(state)
        self.output=[]
        
    
    def update_event(self):
        for event in self.input:
            event.update_event()
        if self.event_type != 'BASIC':
                if self.gate_type == 'AND': # If all inputs are 0, then state = 0, otherwise, state = 1
                    if sum([int(obj.state) for obj in self.input]) == 0:
                        self.state = 0
                    else:
                        self.state = 1 
                elif self.gate_type == 'OR': # If any input is != 0, then state = 0, otherwise, state = 1
                    for obj in self.input:
                        if int(obj.state) != 1:
                            self.state = 0
                            break  
                        else:
                            self.state = 1           
                elif self.gate_type == 'FDEP': # FDEP gates only accepts one input precedence, must combine with and or events prior to input signal.
                    self.state = self.input[0].state
                    #print(self.name,self.input[0].name)
                elif self.gate_type == 'CSP': # Cold Spare currently, only accepts 1 competitor, and 1 spare. Also the 'main' input must come first in the input list.
                    self.main_functioning = self.input[0].state # Is main functioning?
                    self.spare_functioning = self.spare.state # Is Spare functioning?     
                    if self.main_functioning == 1:
                        self.state = 1
                        self.using_spare = 0
                    elif self.main_functioning == 0 and self.competitor.using_spare == 0 and self.spare_functioning == 1:
                        self.state = 1
                        self.using_spare = 1
                    else:
                        self.state = 0
                        self.using_spare = 0

    def event_partial_update(self, new_state):
        
        if self.state == new_state or self.name in Event.red_visited:
            return
        else:
            Event.red_visited.append(self.name)
            self.state = new_state
            Event.count += 1
        for parent in self.output:
            if parent.gate_type == 'AND': # If all inputs are 0, then state = 0, otherwise, state = 1
                if sum([int(obj.state) for obj in parent.input]) == 0:
                    new_state = 0
                else:
                    new_state = 1 
            elif parent.gate_type == 'OR': # If any input is != 0, then state = 0, otherwise, state = 1
                for obj in parent.input:
                    if int(obj.state) != 1:
                        new_state = 0
                        break  
                    else:
                        new_state = 1           
            elif parent.gate_type == 'FDEP': # FDEP gates only accepts one input precedence, must combine with and or events prior to input signal.
                new_state = parent.input[0].state
                #print(parent.name,parent.input[0].name)
            elif parent.gate_type == 'CSP': # Cold Spare currently, only accepts 1 competitor, and 1 spare. Also the 'main' input must come first in the input list.
                parent.main_functioning = parent.input[0].state # Is main functioning?
                parent.spare_functioning = parent.spare.state # Is Spare functioning?     
                if parent.main_functioning == 1:
                    new_state = 1
                    parent.using_spare = 0
                elif parent.main_functioning == 0 and parent.competitor.using_spare == 0 and parent.spare_functioning == 1:
                    new_state = 1
                    parent.using_spare = 1
                else:
                    new_state = 0
                    parent.using_spare = 0
            parent.event_partial_update(new_state)

    def event_partial_update_demo(self, new_state):
        
        if self.state == new_state or self.name in Event.blue_visited:
            return
        else:
            state_to_be_restored = new_state
            Event.blue_visited.append(self.name)
            self.state = new_state
            Event.count += 1
        for parent in self.output:
            if parent.gate_type == 'AND': # If all inputs are 0, then state = 0, otherwise, state = 1
                if sum([int(obj.state) for obj in parent.input]) == 0:
                    new_state = 0
                else:
                    new_state = 1 
            elif parent.gate_type == 'OR': # If any input is != 0, then state = 0, otherwise, state = 1
                for obj in parent.input:
                    if int(obj.state) != 1:
                        new_state = 0
                        break  
                    else:
                        new_state = 1           
            elif parent.gate_type == 'FDEP': # FDEP gates only accepts one input precedence, must combine with and or events prior to input signal.
                new_state = parent.input[0].state
                #print(parent.name,parent.input[0].name)
            elif parent.gate_type == 'CSP': # Cold Spare currently, only accepts 1 competitor, and 1 spare. Also the 'main' input must come first in the input list.
                parent.main_functioning = parent.input[0].state # Is main functioning?
                parent.spare_functioning = parent.spare.state # Is Spare functioning?     
                if parent.main_functioning == 1:
                    new_state = 1
                    parent.using_spare = 0
                elif parent.main_functioning == 0 and parent.competitor.using_spare == 0 and parent.spare_functioning == 1:
                    new_state = 1
                    parent.using_spare = 1
                else:
                    new_state = 0
                    parent.using_spare = 0
            parent.event_partial_update_demo(new_state)
            self.state = state_to_be_restored
    
    

class BasicEvent(Event):
    def __init__(self, name, mttr=None, repair_cost=None, failure_cost=None, initial_state=1):
        super().__init__(name, event_type="BASIC", state=initial_state)
        self.mttr = int(mttr)
        self.repair_cost = int(repair_cost)
        self.failure_cost = int(failure_cost)
        self.repairing = 0
        self.remaining_time_to_repair = 0
    
    def get_repair_status(self):
        return self.repairing
    
    def red_action(self):
        'activate basic event'
        Event.count = 0
        Event.red_visited = []
        self.event_partial_update(0)
        self.remaining_time_to_repair = self.mttr
        return Event.count

    def blue_action(self):
        'inactivate basic event'
        if self.state == 1:
            return 0,[]
        self.repairing = 1
        Event.count = 0
        Event.blue_visited = []
        self.event_partial_update_demo(1)
        return (Event.count, Event.blue_visited)

class IntermediateTopEvent(Event):
    def __init__(self, name,event_type, gate_type=None):
        super().__init__(name, event_type=event_type,state=0)
        self.gate_type=gate_type

class Precedence:
    def __init__(self, source, target, precedence_type, competitor=None):
        self.source = source
        self.target = target
        self.precedence_type = precedence_type
        self.competitor = competitor

class No_Action(Event):
    'No Action class' 
    count = 0
    visited = []
    def __init__(self, name):
        Event.__init__(self, name, 'No Action')
        
    def red_action(self):
        'skip turn'
        return No_Action.count

    def blue_action(self):
        'skip turn'
        return No_Action.count, No_Action.visited