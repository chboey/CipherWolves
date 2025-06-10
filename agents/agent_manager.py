from utils.personas import randomize_personas, randomize_werewolf
from utils.sentiment_ruling import persona_additional_info
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import FunctionTool
from google.adk.sessions import InMemorySessionService
import random
import json
from datetime import datetime
from typing import List, Dict
import asyncio


AGENT_NAME = ['Alice','Bob','Cindy','Dom','Elise']

def web_search(query: str) -> dict:
    """Search the web using Tavily API for additional context about a topic or persona.
    The results of this search are strictly private and should only be used by the agent that initiated the search.
    NOTE: Tracking/logging of agent's search queries must be handled by the caller, not inside this function.
    Args:
        query (str): The search query to find information about
    Returns:
        dict: A dictionary containing the search results with 'status' ('success' or 'error') 
              and either 'answer' with the search results or 'error_message' if an error occurred.
              Note: These results are private to the searching agent and should not be shared.
    """
    try:
        # Log the search query
        # print(f"Search Query: {query}")
        # Check for forbidden search terms
        forbidden_terms = ["werewolf", "imposter", "spy", "traitor", "deception game"]
        if any(term.lower() in query.lower() for term in forbidden_terms):
            # print(f"Forbidden search term detected in query: {query}")
            return {"status": "error", "error_message": "This search query is not allowed"}
        # Use the persona_additional_info function which uses Tavily API
        result = persona_additional_info(query)
        # Log the search result
        # print(f"Search Result: {json.dumps(result, indent=2)}")
        if result and isinstance(result, dict):
            return {"status": "success", "answer": result}
        return {"status": "error", "error_message": "No search results found"}
    except Exception as e:
        # print(f"Search error: {str(e)}")
        return {"status": "error", "error_message": str(e)}

def logged_web_search(query: str, agent_name: str) -> dict:
    """Wrapper around web_search that captures logs while maintaining the original function's behavior.
    
    Args:
        query (str): The search query to find information about
        agent_name (str): The name of the agent performing the search
        
    Returns:
        dict: The original web_search result
    """
    # Get the original result
    result = web_search(query)
    
    # Create a log entry
    log_entry = {
        "query": query,
        "result": result,
        "timestamp": datetime.now().isoformat(),
        "agent": agent_name
    }
    
    # Store the log in a global list
    if not hasattr(logged_web_search, "search_logs"):
        logged_web_search.search_logs = []
    
    # If this is a new round, start a new list of searches
    if not hasattr(logged_web_search, "current_round_searches"):
        logged_web_search.current_round_searches = []
    
    # Add to both the global list and current round list
    logged_web_search.search_logs.append(log_entry)
    logged_web_search.current_round_searches.append(log_entry)
    
    # Broadcast the search query if active_streams is available
    if hasattr(logged_web_search, "active_streams") and logged_web_search.active_streams:
        for game_id, queue in logged_web_search.active_streams.items():
            asyncio.create_task(queue.put({
                "type": "search",
                "data": {
                    "agent": agent_name,
                    "query": query,
                    "result": result
                },
                "timestamp": datetime.now().isoformat()
            }))
    
    return result

def get_search_logs() -> List[Dict]:
    """Get all web search logs.
    
    Returns:
        List[Dict]: List of all web search logs
    """
    return getattr(logged_web_search, "search_logs", [])

def get_current_round_searches() -> List[Dict]:
    """Get all web search logs for the current round.
    
    Returns:
        List[Dict]: List of web search logs for the current round
    """
    return getattr(logged_web_search, "current_round_searches", [])

def clear_search_logs():
    """Clear all web search logs."""
    logged_web_search.search_logs = []

def clear_current_round_searches():
    """Clear the current round's search logs."""
    logged_web_search.current_round_searches = []

def create_web_search_tool(agent_name: str):
    """Create a web search tool for a specific agent.
    
    Args:
        agent_name (str): The name of the agent
        
    Returns:
        FunctionTool: A web search tool configured for the agent
    """
    def agent_web_search(query: str) -> dict:
        return logged_web_search(query, agent_name)
    
    return FunctionTool(func=agent_web_search)

def initialize_agents(num_agents, model="gemini-2.0-flash-001"):
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
        
        # Add strict persona and tool usage instructions
        persona_instruction += """
        You must fully embody your assigned persona at all times and never disclose your personas to other agents,
        regardless of circumstances. If you happen to be the werewolf, you must keep this fact completely hidden from others.
        Under no condition should you search for information related to being a werewolf.
        The web search tool is to be used strictly for gathering information relevant to your persona and its strategies,
        and any findings must remain confidential and consistent with your character.
        All interactions, including searches and communications, must align seamlessly with your assumed identity. Ensure that 
        you only use the keywords provided to you.
        Any information obtained through web searches is strictly private to you and should never be directly shared with other agents.
        """
        
        # Create web search tool for this agent
        web_search_tool = create_web_search_tool(f"agent_{AGENT_NAME[i]}")
        
        # Create agent with the persona and web search tool
        agent = LlmAgent(
            name=f"agent_{AGENT_NAME[i]}",
            model=model,
            description=f"Agent {AGENT_NAME[i]} with a unique persona",
            instruction=persona_instruction,
            input_schema=None,
            tools=[web_search_tool],
            output_key="agent_conversation",
        )
        agents.append(agent)

    return agents, agents[werewolf_index]

def set_active_streams(streams):
    """Set the active streams for broadcasting search queries.
    
    Args:
        streams (dict): Dictionary of game_id to queue mappings
    """
    logged_web_search.active_streams = streams


    
