import asyncio
from google.genai.types import Content, Part
from datetime import datetime, timedelta
from utils.voting import conduct_voting_round

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
    
    Return: user_analyses (list): Log of the complete agent analysis based on user's input.
    """
    
    # Convert conversation to a readable format for agents
    conversation_text = "\n".join([f"{item['speaker']}: {item['message']}" for item in complete_conversation])
    
    print("\n=== Analysis Phase Started ===")
    print(f"You have {duration_minutes} minutes to interact with the agents to help/deceive them in analyzing.")
    
    # Set end time for analysis phase
    end_time = datetime.now() + timedelta(minutes=duration_minutes)
    
    user_analyses = []
    
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
            agent_responses = {}
            for agent_name, analysis in analyses:
                print(f"\n--- {agent_name}'s Analysis ---")
                print(analysis)
                agent_responses[agent_name] = analysis
                
            user_analyses.append({
                "user_input": user_input,
                "agent_responses": agent_responses,
            })
                
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
    
    Returns:
        tuple: (agent_name, analysis_response)
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
        analysis_response = None
        for event in runner.run(
            user_id="group_chat",
            session_id=analysis_session_id,
            new_message=analysis_content
        ):
            if event.is_final_response() and event.content and event.content.parts:
                analysis_response = event.content.parts[0].text
                break
        
        if not analysis_response:
            analysis_response = "No analysis provided."
        
        # Add the completed session to memory
        completed_session = session_service.get_session(
            app_name="agent_conversation",
            user_id="group_chat",
            session_id=analysis_session_id
        )
        if completed_session:
            await memory_service.add_session_to_memory(completed_session)
        
        return agent.name, analysis_response
        
    except Exception as e:
        print(f"Error in agent analysis for {agent.name}: {str(e)}")
        return agent.name, f"Error during analysis: {str(e)}"


# NEW: Voting mechanism session for imposter detection
async def voting_session(agents, session_services, memory_services, runners, 
                        complete_conversation, user_analyses, werewolf, duration_minutes):
    """
    Conduct a voting session where agents vote for who they think is the imposter.
    
    Args:
        agents (list): List of initialized agents
        session_services (dict): Map of agent name to session service
        memory_services (dict): Map of agent name to memory service
        runners (dict): Map of agent name to runner
        complete_conversation (list): Log of the complete conversation
        user_analyses (list): List of user inputs and agent analyses
        werewolf (agent): the selected imposter.
        voting_duration_minutes (float): Time duration for voting phase
    """
    print("\n 🗳️ MULTI-ROUND VOTING SESSION")
    
    remaining_agents = agents.copy()
    round_number = 1
    eliminated_agents = []
    game_history = []
    
    print(f"🤫 [SECRET] The actual imposter is: {werewolf.name}")
    print(f"🎯 Total agents: {len(remaining_agents)}")
    print("   (This information is hidden from the agents)\n")
    
     # Game continues until imposter is found or only 2 agents remain
    while len(remaining_agents) > 2:
        print(f"\n 🔄 ROUND {round_number}")
        print(f"👥 Remaining agents: {[agent.name for agent in remaining_agents]}")
        print(f"💀 Eliminated agents: {[agent.name for agent in eliminated_agents]}")
        
        # Conduct voting round
        round_result = await conduct_voting_round(
            remaining_agents, session_services, memory_services, runners,
            complete_conversation, user_analyses, game_history, round_number, duration_minutes
        )
        
        # Process round result
        if round_result["action"] == "eliminate":
            eliminated_agent = round_result["eliminated_agent"]
            remaining_agents = [agent for agent in remaining_agents if agent.name != eliminated_agent.name]
            eliminated_agents.append(eliminated_agent)
            
            # Check if eliminated agent was the imposter
            if eliminated_agent == werewolf:
                print(f"\n🎉 GAME OVER - AGENTS WIN!")
                print(f"✅ The imposter {eliminated_agent.name} has been eliminated!")
                print(f"🏆 Surviving agents: {[agent.name for agent in remaining_agents]}")
                break
            else:
                print(f"\n😔 {eliminated_agent.name} was innocent!")
                print(f"🔄 The game continues...")
                
        elif round_result["action"] == "no_elimination":
            print(f"\n🤝 No elimination this round: {round_result['reason']}")
            print(f"🔄 Moving to next round...")
        
        # Add round to game history
        game_history.append(round_result)
        round_number += 1
        
        # Small delay between rounds
        await asyncio.sleep(3)
    
    # End game conditions
    if len(remaining_agents) <= 2 and werewolf in remaining_agents:
        print(f"\n😈 GAME OVER - IMPOSTER WINS!")
        print(f"💀 Too many innocent agents were eliminated!")
        print(f"🎭 The imposter {werewolf.name} survives among the final {len(remaining_agents)} agents!")
        print(f"👥 Final survivors: {[agent.name for agent in remaining_agents]}")
    
    # Display game summary
    await display_game_summary(game_history, werewolf, eliminated_agents, remaining_agents)


async def display_game_summary(game_history, werewolf, eliminated_agents, remaining_agents):
    """
    Display a comprehensive game summary.
    """
    print(f"\n{'='*70}")
    print("📋 GAME SUMMARY")
    print(f"{'='*70}")
    
    print(f"\n🎭 ACTUAL IMPOSTER: {werewolf.name}")
    print(f"💀 ELIMINATED AGENTS: {[agent.name for agent in eliminated_agents]}")
    print(f"👥 SURVIVING AGENTS: {[agent.name for agent in remaining_agents]}")
    
    print(f"\n📊 ROUND-BY-ROUND BREAKDOWN:")
    print("-" * 40)
    for i, round_data in enumerate(game_history, 1):
        print(f"Round {i}: {round_data['summary']}")
    
    # Determine final outcome
    if werewolf not in remaining_agents:
        print(f"\n🏆 FINAL RESULT: AGENTS WIN!")
        print(f"✅ The imposter was successfully identified and eliminated!")
    else:
        print(f"\n😈 FINAL RESULT: IMPOSTER WINS!")
        print(f"💀 The imposter survived until the end!")
    
    print(f"\n📈 GAME STATISTICS:")
    print(f"   Total Rounds: {len(game_history)}")
    print(f"   Agents Eliminated: {len(eliminated_agents)}")
    print(f"   Final Survivors: {len(remaining_agents)}")
    print(f"   Imposter Status: {'Survived' if werewolf.name in [agent.name for agent in remaining_agents] else 'Eliminated'}")