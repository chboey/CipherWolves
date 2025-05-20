from utils.personas import randomize_personas
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

AGENT_NAME = ['Alice','Bob','Cindy','Dom','Elise']


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
        agent = LlmAgent(
            name=f"agent_{AGENT_NAME[i]}",
            model=LiteLlm(model=model),
            description=f"Agent {AGENT_NAME[i]} with a unique persona",
            instruction=persona_instruction,
            input_schema=None,
            tools=[],
            output_key="agent_conversation",
        )
        agents.append(agent)
    
    return agents


    
