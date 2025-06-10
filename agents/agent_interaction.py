import asyncio
from google.genai.types import Content, Part
from datetime import datetime, timedelta
from utils.voting import conduct_voting_round
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.runners import Runner
import colorful as cf

async def call_conversationFlow(agents, keywords, duration_minutes):
    """
    Run a conversation between agents for a specified duration using only allowed keywords.
    
    Args:
        agents (list): List of initialized agents
        keywords (list): List of allowed keywords
        duration_minutes (float): Duration of the conversation in minutes
        
    Returns:
        tuple: (session_services, memory_services, runners, complete_conversation)
    """
    # Initialize services for each agent
    session_services = {}
    memory_services = {}
    runners = {}
    
    # Initialize services for each agent
    for agent in agents:
        session_services[agent.name] = InMemorySessionService()
        memory_services[agent.name] = InMemoryMemoryService()
        runners[agent.name] = Runner(
            agent=agent,
            app_name="agent_conversation",
            session_service=session_services[agent.name],
            memory_service=memory_services[agent.name]
        )
    
    # Create initial session for each agent
    sessions = {}
    for agent in agents:
        sessions[agent.name] = session_services[agent.name].create_session(
            app_name="agent_conversation",
            user_id="group_chat",
            session_id=f"session_{agent.name}"
        )
        
    # Storing complete conversation for tracking.
    complete_conversation = []
    
    # Set end time
    end_time = datetime.now() + timedelta(minutes=duration_minutes)
    
    # Initial message to start conversation
    current_message = f"Hello everyone! Let's discuss these keywords: {', '.join(keywords)}. What are your thoughts?"
    current_speaker = agents[0].name

    # Logging setup - conversation start, allowed keywords.
    # print(f"\n=== Conversation Started at {datetime.now().strftime('%H:%M:%S')} ===\n")
    # print(f"Allowed Keywords: {', '.join(keywords)}\n")
    
    
    while datetime.now() < end_time:
        # Get the next speaker (round-robin)
        next_speaker_idx = (agents.index(next(a for a in agents if a.name == current_speaker)) + 1) % len(agents)
        next_speaker = agents[next_speaker_idx]
        
        # Create message content
        message_content = Content(
            parts=[Part(text=f"[{current_speaker} says]: {current_message}")],
            role="user"
        )
        
        # Get response from next speaker
        print(f"\n{current_speaker} says: {current_message}")
        
        final_response = None
        for event in runners[next_speaker.name].run(
            user_id="group_chat",
            session_id=f"session_{next_speaker.name}",
            new_message=message_content
        ):
            if event.is_final_response() and event.content and event.content.parts:
                final_response = event.content.parts[0].text
                break
        
        if final_response:
            current_message = final_response
            current_speaker = next_speaker.name
            
            # Append into conversation logs.
            complete_conversation.append({"speaker": current_speaker, "message": current_message})
            
            # Add the completed session to memory
            completed_session = session_services[next_speaker.name].get_session(
                app_name="agent_conversation",
                user_id="group_chat",
                session_id=f"session_{next_speaker.name}"
            )
            await memory_services[next_speaker.name].add_session_to_memory(completed_session)
        else:
            current_message = "I don't have a response at this moment."
            current_speaker = next_speaker.name
        
        # Small delay between messages
        await asyncio.sleep(2)

    # print(f"\n=== Conversation Ended at {datetime.now().strftime('%H:%M:%S')} ===\n")
    
    return session_services, memory_services, runners, complete_conversation

async def user_interaction(agents, session_services, memory_services, runners, complete_conversation, duration_minutes):
    """
    Process user input and get agent analyses. This function is now API-driven but maintains terminal logging.
    
    Args:
        agents (list): List of initialized agents
        session_services (dict): Map of agent name to session service
        memory_services (dict): Map of agent name to memory service
        runners (dict): Map of agent name to runner
        complete_conversation (list): Log of the complete conversation
        duration_minutes (float): Time duration for analysis phase
    
    Return: user_analyses (list): Log of the complete agent analysis based on user's input.
    """
    
    # Convert conversation to a readable format for agents
    conversation_text = "\n".join([f"{item['speaker']}: {item['message']}" for item in complete_conversation])
    
    # print("\n=== Analysis Phase Started ===")
    # print(f"Analysis phase duration: {duration_minutes} minutes")
    
    # Set end time for analysis phase
    end_time = datetime.now() + timedelta(minutes=duration_minutes)
    
    user_analyses = []
    
    # Process each agent's analysis sequentially
    # print("\n=== Agents Analyzing ===")
    agent_responses = {}
    for agent in agents:
        # print(f"\n--- {agent.name} is analyzing... ---")
        final_response = None
        async for agent_name, response in get_agent_analysis(
            agent, 
            runners[agent.name], 
            session_services[agent.name], 
            memory_services[agent.name], 
            conversation_text
        ):
            # print(f"\n{agent_name}'s Analysis (intermediate):")
            # print(response)
            final_response = response
            
        if final_response:
            agent_responses[agent_name] = final_response
            # print(f"\n{agent_name}'s Final Analysis:")
            # print(final_response)
            
    user_analyses.append({
        "user_input": "Analysis phase",
        "agent_responses": agent_responses,
    })
    
    # print("\n=== Analysis Phase Completed ===")
    return user_analyses

async def get_agent_analysis(agent, runner, session_service, memory_service, analysis_prompt):
    """
    Get analysis from a single agent. Should return analysis response, and vote or not (not compulsory).
    
    Args:
        agent: The agent to get analysis from
        runner: The runner for the agent
        session_service: The session service for the agent
        memory_service: The memory service for the agent
        analysis_prompt: The prompt for analysis
    
    Yields:
        tuple: (agent_name, analysis_response) for intermediate results
    """
    # Create analysis content
    analysis_content = Content(
        parts=[Part(text=analysis_prompt)],
        role="user"
    )
    
    # Create a new session for this analysis
    analysis_session_id = f"session_{agent.name}"
    
    try:
        # Get existing session or create new one
        session = session_service.get_session(
            app_name="agent_conversation",
            user_id="group_chat",
            session_id=analysis_session_id
        )
        
        if not session:
            # Create new session if it doesn't exist
            session = session_service.create_session(
                app_name="agent_conversation",
                user_id="group_chat",
                session_id=analysis_session_id,
                state={"status": "initialized"}
            )
            
        if not session:
            raise ValueError(f"Failed to create/get session: {analysis_session_id}")
            
        # Get analysis from agent
        final_response = None
        current_response = ""
        
        for event in runner.run(
            user_id="group_chat",
            session_id=analysis_session_id,
            new_message=analysis_content
        ):
            if event.content and event.content.parts:
                response_text = event.content.parts[0].text
                
                # Only yield if we have new content
                if response_text != current_response:
                    current_response = response_text
                    yield agent.name, response_text
                
                if event.is_final_response():
                    final_response = response_text
        
        if not final_response:
            final_response = "No analysis provided."
        
        # Add the completed session to memory
        completed_session = session_service.get_session(
            app_name="agent_conversation",
            user_id="group_chat",
            session_id=analysis_session_id
        )
        if completed_session:
            await memory_service.add_session_to_memory(completed_session)
        
        # Yield final response if it's different from the last intermediate response
        if final_response != current_response:
            yield agent.name, final_response
        
    except Exception as e:
        # print(f"Error in agent analysis for {agent.name}: {str(e)}")
        yield agent.name, f"Error during analysis: {str(e)}"


# NEW: Voting mechanism session for imposter detection
async def voting_session(agents, session_services, memory_services, runners, 
                        complete_conversation, user_analyses, werewolf):
    """
    Conduct a single voting round where agents vote for who they think is the imposter.
    
    Args:
        agents (list): List of initialized agents
        session_services (dict): Map of agent name to session service
        memory_services (dict): Map of agent name to memory service
        runners (dict): Map of agent name to runner
        complete_conversation (list): Log of the complete conversation
        user_analyses (list): List of user inputs and agent analyses
        werewolf (agent): the selected imposter.
        
    Returns:
        dict: Complete voting results including votes, counts, and elimination details
    """
    # Conduct single voting round
    round_result = await conduct_voting_round(
        agents, session_services, memory_services, runners,
        complete_conversation, user_analyses, [], 1
    )
    
    # Process round result
    if round_result["action"] == "eliminate":
        eliminated_agent = round_result["eliminated_agent"]
        return round_result
        
    elif round_result["action"] == "no_elimination":
        return round_result


async def display_game_summary(game_history, werewolf, eliminated_agents, remaining_agents):
    """
    Display a comprehensive game summary.
    """
    # print(f"\n{'='*70}")
    # print("📋 GAME SUMMARY")
    # print(f"{'='*70}")
    
    # print(f"\n🎭 ACTUAL IMPOSTER: {werewolf.name}")
    # print(f"💀 ELIMINATED AGENTS: {[agent.name for agent in eliminated_agents]}")
    # print(f"👥 SURVIVING AGENTS: {[agent.name for agent in remaining_agents]}")
    
    # print(f"\n📊 ROUND-BY-ROUND BREAKDOWN:")
    # print("-" * 40)
    for i, round_data in enumerate(game_history, 1):
        # print(f"Round {i}: {round_data['summary']}")
        pass
    
    # Determine final outcome
    if werewolf not in remaining_agents:
        # print(f"\n🏆 FINAL RESULT: AGENTS WIN!")
        # print(f"✅ The imposter was successfully identified and eliminated!")
        pass
    else:
        # print(f"\n😈 FINAL RESULT: IMPOSTER WINS!")
        # print(f"💀 The imposter survived until the end!")
        pass
    
    # print(f"\n📈 GAME STATISTICS:")
    # print(f"   Total Rounds: {len(game_history)}")
    # print(f"   Agents Eliminated: {len(eliminated_agents)}")
    # print(f"   Final Survivors: {len(remaining_agents)}")
    # print(f"   Imposter Status: {'Survived' if werewolf.name in [agent.name for agent in remaining_agents] else 'Eliminated'}")

async def run_game_round(agents, session_services, memory_services, runners, werewolf, duration_minutes):
    """
    Run a complete game round consisting of user interaction phase followed by voting phase.
    """
    # print("\n" + "="*50)
    # print(f"🎮 STARTING NEW ROUND")
    # print("="*50)
    
    # Initialize empty conversation and analyses for this round
    complete_conversation = []
    user_analyses = []
    
    # Phase 1: User Interaction
    # print("\n📝 PHASE 1: USER INTERACTION")
    user_analyses = await user_interaction(
        agents, 
        session_services, 
        memory_services, 
        runners, 
        complete_conversation, 
        duration_minutes
    )
    
    # If no analyses were performed, end the game
    if not user_analyses:
        # print("\n⚠️ No analyses were performed. Ending game.")
        return False, [], [], []
    
    # Phase 2: Voting
    # print("\n🗳️ PHASE 2: VOTING")
    round_result = await voting_session(
        agents,
        session_services,
        memory_services,
        runners,
        complete_conversation,
        user_analyses,
        werewolf
    )
    
    # Check if werewolf is still in the game
    if round_result["action"] == "eliminate":
        eliminated_agent = round_result["eliminated_agent"]
        # print("\n🎉 GAME OVER - AGENTS WIN!")
        return False, [], [agent for agent in agents if agent.name != eliminated_agent], [agent for agent in agents if agent.name == eliminated_agent]
    
    return True, [], [agent for agent in agents if agent.name != eliminated_agent], [agent for agent in agents if agent.name == eliminated_agent]

async def play_game(agents, session_services, memory_services, runners, werewolf, duration_minutes):
    """
    Main game loop that manages the complete game flow.
    """
    # print("\n🎮 WEREWOLF GAME STARTED")
    # print(f"🎭 The imposter is: {werewolf.name} (This is hidden from the agents)")
    
    round_number = 1
    game_continues = True
    game_history = []
    eliminated_agents = []
    remaining_agents = agents.copy()
    
    while game_continues:
        game_continues, round_history, eliminated_agents, remaining_agents = await run_game_round(
            agents,
            session_services,
            memory_services,
            runners,
            werewolf,
            duration_minutes
        )
        round_number += 1
    
    # Display final game summary
    await display_game_summary(game_history, werewolf, eliminated_agents, remaining_agents)