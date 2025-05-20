from agents.agent_manager import initialize_agents
from utils.keywords import generate_keywords
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.memory import InMemoryMemoryService
from google.genai.types import Content, Part
import asyncio
from datetime import datetime, timedelta

async def run_conversation(agents, keywords, duration_minutes=0.5):
    """
    Run a conversation between agents for a specified duration using only allowed keywords.
    
    Args:
        agents (list): List of initialized agents
        keywords (list): List of allowed keywords
        duration_minutes (int): Duration of the conversation in minutes
    """
    # Initialize services for each agent
    session_services = {}
    memory_services = {}
    runners = {}
    
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
    
    # Set end time
    end_time = datetime.now() + timedelta(minutes=duration_minutes)
    
    # Initial message to start conversation
    current_message = f"Hello everyone! Let's discuss these keywords: {', '.join(keywords)}. What are your thoughts?"
    current_speaker = agents[0].name
    
    print(f"\n=== Conversation Started at {datetime.now().strftime('%H:%M:%S')} ===\n")
    print(f"Allowed Keywords: {', '.join(keywords)}\n")
    
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
    
    print(f"\n=== Conversation Ended at {datetime.now().strftime('%H:%M:%S')} ===\n")

if __name__ == "__main__":
    # Generate 100 random keywords
    keywords = generate_keywords(100)
    
    # Initialize 5 agents
    agents = initialize_agents(5)
    
    # Add keyword restriction to each agent's instruction
    for agent in agents:
        keyword_instruction = (
            f"\n\nIMPORTANT: You can ONLY use these keywords in your responses: {', '.join(keywords)}. "
            "You must incorporate at least one keyword in each response. "
            "Be creative in how you use these keywords while maintaining your persona."
        )
        agent.instruction += keyword_instruction
    
    # Print agent information
    for i, agent in enumerate(agents, 1):
        print(f"\nAgent {i}:")
        print(f"Name: {agent.name}")
        print(f"Description: {agent.description}")
        print("Instruction:", agent.instruction[:100] + "..." if len(agent.instruction) > 100 else agent.instruction)
    
    # Run the conversation
    asyncio.run(run_conversation(agents, keywords, duration_minutes=0.5))