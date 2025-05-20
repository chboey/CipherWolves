import asyncio
from google.genai.types import Content, Part
from datetime import datetime, timedelta

async def user_interaction(agents, session_services, memory_services, runners, complete_conversation, duration_minutes):
    """
    users can directly interact with the agents, deceiving or helping the agents. The user would be directly communicating with all of the agents
    
     Args:
        agents (list): List of initialized agents
        session_services (dict): Map of agent name to session service
        memory_services (dict): Map of agent name to memory service
        runners (dict): Map of agent name to runner
        complete_conversation (list): Log of the complete conversation
        duration_minutes (float): Time duration for analysis phase
    """
    
    # Convert conversation to a readable format for agents
    conversation_text = "\n".join([f"{item['speaker']}: {item['message']}" for item in complete_conversation])
    
    print("\n=== Analysis Phase Started ===")
    print(f"You have {duration_minutes} minutes to interact with the agents to help/deceive them in analyzing.")
    
    # Set end time for analysis phase
    end_time = datetime.now() + timedelta(minutes=duration_minutes)
    
    while datetime.now() < end_time:
        # Wait for user input or timeout
        try:
            # User input in duration_minutes
            user_input = input("Enter your input for the agents to discuss: ")
            
            # Skip empty input
            if not user_input.strip():
                continue
                
            # Prepare analysis prompt with conversation context and user question
            analysis_prompt = f"""
            Here is the complete conversation that occurred:
            
            {conversation_text}
            
            The user is now asking: "{user_input}"
            
            Please analyze this question in the context of the previous conversation 
            and provide your thoughts, insights, or vote if requested.
            """
            
            print("\n=== Agents Analyzing ===")
            
            # Get analysis from each agent simultaneously
            analysis_tasks = []
            for agent in agents:
                analysis_tasks.append(
                    get_agent_analysis(
                        agent, 
                        runners[agent.name], 
                        session_services[agent.name], 
                        memory_services[agent.name], 
                        analysis_prompt
                    )
                )
            
            # Wait for all analysis to complete
            analyses = await asyncio.gather(*analysis_tasks)
            
            # Display all analysis.
            for agent_name, analysis in analyses:
                print(f"\n--- {agent_name}'s Analysis ---")
                print(analysis)
                
        except asyncio.TimeoutError:
            # No input received within timeout period, continue the loop
            remaining_time = (end_time - datetime.now()).total_seconds() / 60
            if remaining_time > 0:
                print(f"\rTime remaining: {remaining_time:.1f} minutes", end="")
            continue
        except Exception as e:
            print(f"Error: {e}")
            break
    
    print("\n=== Analysis Phase Ended ===")
    

async def get_agent_analysis(agent, runner, session_service, memory_service, analysis_prompt):
    """
    Get analysis from a single agent. Should return analysis response, and vote or not (not compulsory).
    
    Args:
        agent: The agent to get analysis from
        runner: The runner for the agent
        session_service: The session service for the agent
        memory_service: The memory service for the agent
        analysis_prompt: The prompt for analysis
    
    Returns:
        tuple: (agent_name, analysis_response)
    """
    # Create analysis content
    analysis_content = Content(
        parts=[Part(text=analysis_prompt)],
        role="user"
    )
    
    # Create a new session for this analysis
    analysis_session_id = f"analysis_{agent.name}_{datetime.now().strftime('%H%M%S')}"
    session_service.create_session(
        app_name="agent_analysis",
        user_id="analysis",
        session_id=analysis_session_id
    )
    
    # Get analysis from agent
    analysis_response = None
    for event in runner.run(
        user_id="analysis",
        session_id=analysis_session_id,
        new_message=analysis_content
    ):
        if event.is_final_response() and event.content and event.content.parts:
            analysis_response = event.content.parts[0].text
            break
        
        # Future have to get this analysis log for post-analysis Insight...
    
    if not analysis_response:
        analysis_response = "No analysis provided."
    
    # Add the completed session to memory
    completed_session = session_service.get_session(
        app_name="agent_analysis",
        user_id="analysis",
        session_id=analysis_session_id
    )
    await memory_service.add_session_to_memory(completed_session)
    
    return agent.name, analysis_response


async def agent_voting(): 
    
    return 