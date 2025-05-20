from utils.keywords import generate_keywords
from utils.personas import randomize_personas
from google.adk.agents import Agent


def initialize_agents(num_agents, model="gemini-2.0-flash"):
    """
    Initialize n number of agents with random personas.
    
    Args:
        num_agents (int): Number of agents to initialize
        model (str): The model to use for the agents
    
    Returns:
        list: List of initialized agents
    """
    agents = []
    for i in range(num_agents):
        # Get random persona instruction
        persona_instruction = randomize_personas()
        
        # Create agent with the persona
        agent = Agent(
            name=f"agent_{i+1}",
            model=model,
            description=f"Agent {i+1} with a unique persona",
            instruction=persona_instruction,
            tools=[]
        )
        agents.append(agent)
    
    return agents


    
