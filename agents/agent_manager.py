from utils.personas import randomize_personas, randomize_werewolf
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
import random

AGENT_NAME = ['Alice','Bob','Cindy','Dom','Elise']


def initialize_agents(num_agents, model="gemini-2.0-flash"):
    """
    Initialize n number of agents with random personas.
    One random agent will be selected as the werewolf.
    
    Args:
        num_agents (int): Number of agents to initialize
        model (str): The model to use for the agents
    
    Returns:
        list: List of initialized agents
    """
    agents = []
    # Randomly select one agent to be the werewolf
    werewolf_index = random.randint(0, num_agents - 1)
    
    for i in range(num_agents):
        # Get random persona instruction
        persona_instruction = randomize_personas()
        
        # If this is the werewolf agent, add werewolf instruction
        if i == werewolf_index:
            werewolf_instruction = randomize_werewolf()
            persona_instruction += f"\n{werewolf_instruction}"
        
        # Create agent with the persona
        agent = LlmAgent(
            name=f"agent_{AGENT_NAME[i]}",
            model=model,
            description=f"Agent {AGENT_NAME[i]} with a unique persona",
            instruction=persona_instruction,
            input_schema=None,
            tools=[],
            output_key="agent_conversation",
        )
        agents.append(agent)
    
    return agents, agents[werewolf_index]


    
