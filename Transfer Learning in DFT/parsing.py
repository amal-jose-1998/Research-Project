import xml.etree.ElementTree as ET
from system import System
from element import BasicEvent,IntermediateTopEvent,Precedence

class Parse:
    '''
    Simple model parse interface
    '''
 
    def from_file(file_name):
        file_type = file_name.split('.')[-1]                    # Gets the file_type 
        if file_type == 'xml':                                  # If xml
            system = Parse_XML.parse_xml(file_name)             # Use xml parser
            return system
    # Other file_type parsers are called here.

class Parse_XML():
    '''
    XML parse interface
    '''

    def parse_xml(xml_file):                                   
        tree = ET.parse(xml_file)
        root = tree.getroot()

        system = System()

        # Parse events
        for event_elem in root.findall('.//event'):
            name = event_elem.get('name')
            event_type = event_elem.get('type')
            gate_type = event_elem.get('gate_type')
            mttr = event_elem.get('mttr')
            repair_cost = event_elem.get('repair_cost')
            failure_cost = event_elem.get('failure_cost')
            initial_state = event_elem.get('initial_state')

            if event_type == 'BASIC':
                event = BasicEvent(name, mttr, repair_cost, failure_cost, initial_state)
            else:
                event = IntermediateTopEvent(name, event_type, gate_type)

            system.add_event(event)

        # Parse precedences
        for precedence_elem in root.findall('.//precedence'):
            source = precedence_elem.get('source')
            target = precedence_elem.get('target')
            precedence_type = precedence_elem.get('type')
            competitor = precedence_elem.get('competitor')

            precedence = Precedence(source, target, precedence_type, competitor)
            system.add_precedence(precedence)

        return system
    