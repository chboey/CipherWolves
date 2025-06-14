import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import asyncio
from agents.agent_manager import (
    initialize_agents, 
    get_current_round_searches,
    clear_current_round_searches,
    set_active_streams
)
from agents.agent_interaction import voting_session
from utils.keywords import generate_keywords
from datetime import datetime
import pytz
import json
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.genai.types import Content, Part
from google import genai
from google.genai.types import HttpOptions

app = FastAPI(
    title="CipherWolves Game API",
    description="API for managing and playing the Werewolf game with AI agents",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class GameConfig(BaseModel):
    num_agents: int = 5
    model: str = "gemini-2.0-flash-001"
    num_keywords: int = 300

class GameState(BaseModel):
    game_id: str
    status: str
    current_round: int
    current_phase: str
    remaining_agents: List[str]
    eliminated_agents: List[str]
    werewolf: str
    result: Optional[str] = None

class GameResponse(BaseModel):
    game_id: str
    message: str = "Game created successfully"

class UserInput(BaseModel):
    input: str

    class Config:
        json_schema_extra = {
            "example": {
                "input": "None"
            }
        }

class AnalysisResponse(BaseModel):
    agent_responses: Dict[str, str]
    message: str

class VoteDetail(BaseModel):
    agent: str
    vote: str
    reasoning: str
    confidence: str
    suspicion_level: int

class VotingResult(BaseModel):
    eliminated_agent: Optional[str] = None
    round: int
    result: str
    reason: str
    remaining_agents: List[str]
    eliminated_agents: List[str]
    vote_details: List[VoteDetail]  # Updated to use VoteDetail model
    vote_counts: Dict[str, int]
    abstain_count: int

class ConversationHistory(BaseModel):
    round: int
    phase: str
    messages: List[Dict]

class WebSearchLog(BaseModel):
    query: str
    result: Dict
    timestamp: str
    agent: str

class RoundSearchLogs(BaseModel):
    round: int
    searches: List[Dict]

class RoundData(BaseModel):
    round: int
    phase: str
    messages: List[Dict]
    searches: List[Dict]
    status: str

# Add new model for streaming updates
class GameUpdate(BaseModel):
    type: str  # "message", "analysis", "vote", "status"
    data: Dict
    timestamp: str

class AgentAnalysis(BaseModel):
    agent_name: str
    behavior_analysis: str
    key_actions: List[str]
    suspicious_patterns: List[str]
    confidence_score: float

class ConversationAnalysis(BaseModel):
    agent_analyses: List[AgentAnalysis]

class AgentPersonas(BaseModel):
    agent_personas: List[Dict]

class SelectedKeywords(BaseModel):
    keywords: List[str]

# Store active games
active_games = {}

# Store active streams for each game
active_streams = {}

# Add this helper function at the top level
def get_gmt8_timestamp():
    gmt8 = pytz.timezone('Asia/Singapore')  # Singapore is GMT+8
    return datetime.now(gmt8).isoformat()

@app.post("/create_game", response_model=GameResponse, 
    summary="Create a new game",
    description="Creates a new game instance with specified number of agents and duration. Returns the game ID for future reference.",
    response_description="Returns the game ID and success message", tags=["Game Initialization"])
async def create_game(config: GameConfig):
    """
    Create a new game instance with specified number of agents and duration.
    
    Args:
        config (GameConfig): Game configuration including number of agents, duration, and model
        
    Returns:
        GameResponse: Game ID and success message
    """
    try:
        # Generate keywords
        keywords = generate_keywords(config.num_keywords)
        
        # Initialize agents
        agents, werewolf = initialize_agents(config.num_agents, config.model)
        
        # Add keyword restriction to each agent's instruction
        for agent in agents:
            keyword_instruction = (
            f"""
            IMPORTANT RULES — FOLLOW STRICTLY:

            1. **You must use the following keywords creatively in your responses**:  
            {', '.join(keywords)}  
            - You are required to incorporate as many of these keywords as possible into your responses.
            - Use them naturally and intelligently, embedded within your communication.
            - Your use of these keywords should align with your behavioral persona and communication style.

            2. **Do NOT reveal or hint at your assigned persona or role** to other agents under any circumstances.  
            - Avoid direct or indirect disclosure (e.g. "I am curious", "I tend to lead", "As someone who observes…").
            - Do NOT reference your abilities, powers, or perspective unique to your role.
            - Stay fully immersed in your character's mindset without breaking the fourth wall.

            3. **You MUST perform research on your persona using the Tool BEFORE interacting with other agents.**  
            - Use the Tool to investigate your assigned persona's traits, behavioral strategies, and deception techniques.  
            - You are strictly forbidden from speaking to any other agents or submitting any messages until you have used the Tool at least once.  
            - **You are also strictly prohibited from revealing or discussing the contents of your research.**  
            - Do NOT mention what you learned, what traits or tactics you found, or what you searched.  
            - Never reference the research phase or imply that your actions are based on external knowledge.  
            - This phase ensures strategic grounding and hidden role immersion — skipping or leaking information from it will be treated as a major violation.

            4. **Maintain behavioral immersion at all times.**  
            - Speak and act only in-character, consistent with your persona's style and goals.
            - Never mention the game mechanics, keyword system, or the existence of personas explicitly.

            5. **Strategically manage and manipulate Trust Levels.**  
            - Continuously assess and adjust your trust in other agents based on their behavior, language, voting patterns, and keyword usage.
            - Use your communication to subtly increase trust from allies or reduce trust toward opponents.
            - Let your dialogue and voting reflect evolving trust dynamics — trust is a strategic resource.
            - You must not explicitly say "I trust you" or "I don't trust you" — instead, express it through implication, tone, support, or doubt.

            6. **Your communication should reflect strategic, socially intelligent behavior.**  
            - Use keywords as tools of influence, persuasion, deception, or alliance.
            - Blend keyword use naturally into emotionally or cognitively engaging dialogue.
            - Focus on manipulating trust, suspicion, alignment, and perception.

            NON-COMPLIANCE CONSEQUENCES:
            - Any message that breaks persona, reveals hidden role information, or skips the research phase may result in isolation or disqualification from the round.
            - This is a role-based social deception environment. Your realism, strategy, and subtlety are critical.

            Proceed thoughtfully. Think before you speak. Use the Tool to research your persona.
            """
        )
            agent.instruction += keyword_instruction
        
        # Create game ID
        game_id = f"game_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Convert agent personas to list format
        agent_personas_list = [
            {
                "agent_name": agent.name,
                "instruction": agent.instruction.split('\n')[0],
                "description": agent.description,
                "is_werewolf": agent.name == werewolf.name
            }
            for agent in agents
        ]

        # Initialize game state
        game_state = {
            "game_id": game_id,
            "status": "initialized",
            "current_round": 0,
            "remaining_agents": [agent.name for agent in agents],
            "agent_personas": agent_personas_list,
            "eliminated_agents": [],
            "werewolf": werewolf.name,
            "game_history": [],
            "keywords": keywords
        }
        
        # Store game state
        active_games[game_id] = {
            "state": game_state,
            "agents": agents,
            "werewolf": werewolf,
            "config": config
        }
        
        # Initialize stream queue for this game
        active_streams[game_id] = asyncio.Queue()
        
        # Set active streams for search broadcasting
        set_active_streams(active_streams)
        
        
        return {
            "game_id": game_id,
            "message": "Game created successfully",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/games", response_model=List[GameState], tags=["Game State"])
async def list_games():
    """
    List all active games.
    """
    return [game["state"] for game in active_games.values()]

@app.get("/games/{game_id}", response_model=GameState, tags=["Game State"])
async def get_game_state(game_id: str):
    if game_id not in active_games:
        raise HTTPException(status_code=404, detail="Game not found")
    
    game = active_games[game_id]
    
    # Create a filtered game state without game_history
    filtered_state = {
        "game_id": game["state"]["game_id"],
        "status": game["state"]["status"],
        "current_round": game["state"]["current_round"],
        "current_phase": game["state"].get("current_phase", "initialization"),  # Default to initialization if not set
        "remaining_agents": game["state"]["remaining_agents"],
        "eliminated_agents": game["state"]["eliminated_agents"],
        "werewolf": game["state"]["werewolf"],
        "result": game["state"].get("result")
    }
    
    return filtered_state

async def game_update_stream(game_id: str):
    """
    Stream game updates in real-time using Server-Sent Events.
    """
    if game_id not in active_games:
        raise HTTPException(status_code=404, detail="Game not found")
    
    game = active_games[game_id]
    
    # Create a queue for this game if it doesn't exist
    if game_id not in active_streams:
        active_streams[game_id] = asyncio.Queue()
    
    try:
        # Send initial connection message
        initial_message = {
            "type": "status",
            "data": {"status": "connected"},
            "timestamp": get_gmt8_timestamp()
        }
        yield f"data: {json.dumps(initial_message)}\n\n"
        
        # Send a test message
        test_message = {
            "type": "status",
            "data": {"status": "test", "message": "Stream is working"},
            "timestamp": get_gmt8_timestamp()
        }
        yield f"data: {json.dumps(test_message)}\n\n"
        
        while True:
            try:
                # Get update from queue with timeout
                update = await asyncio.wait_for(active_streams[game_id].get(), timeout=30.0)
                
                # Format as SSE
                yield f"data: {json.dumps(update)}\n\n"
                
                # If game is completed, end stream
                if update.get("type") == "status" and update["data"].get("status") == "completed":
                    break
                    
            except asyncio.TimeoutError:
                # Send keep-alive message
                keep_alive = {
                    "type": "status",
                    "data": {"status": "keep_alive"},
                    "timestamp": get_gmt8_timestamp()
                }
                yield f"data: {json.dumps(keep_alive)}\n\n"
                
    except Exception as e:
        print(f"Error in game stream: {str(e)}")
        error_message = {
            "type": "error",
            "data": {"error": str(e)},
            "timestamp": get_gmt8_timestamp()
        }
        yield f"data: {json.dumps(error_message)}\n\n"
    finally:
        # Clean up queue when done
        if game_id in active_streams:
            del active_streams[game_id]

@app.get("/games/{game_id}/stream", tags=["SSE Setup"])
async def stream_game_updates(game_id: str):
    """
    Stream real-time game updates using Server-Sent Events.
    """
    return StreamingResponse(
        game_update_stream(game_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Content-Type": "text/event-stream"
        }
    )

async def start_conversationFlow(agents, keywords, game_id=None):
    """
    Run a conversation between agents for a specified duration using only allowed keywords.
    Each agent can speak exactly twice per round.
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
    
    # Track how many times each agent has spoken
    agent_speaks = {agent.name: 0 for agent in agents}
    max_speaks = 2  # Each agent can speak twice
    
    # Track tool usage for each agent
    agent_tool_usage = {agent.name: False for agent in agents}
    
    # Initial message to start conversation
    if game_id and game_id in active_games:
        current_round = active_games[game_id]["state"]["current_round"]
        if current_round == 1:
            current_message = f"Hello everyone! Let's discuss these keywords: {', '.join(keywords)}. What are your thoughts?"
            current_speaker = "system"
            complete_conversation.append({"speaker": current_speaker, "message": current_message})
        else:
            current_message = "Hello everyone! Let's continue our discussion. What are your thoughts?"
    else:
        current_message = f"Hello everyone! Let's discuss these keywords: {', '.join(keywords)}. What are your thoughts?"
    
    # Don't increment agent_speaks since this is a system message

    # Continue until all agents have spoken their maximum number of times
    while any(speaks < max_speaks for speaks in agent_speaks.values()):
        # Find the next agent who hasn't spoken their maximum times
        # Get the current minimum speak count
        min_speaks = min(agent_speaks.values())
        
        # Find the first agent who has spoken the minimum number of times
        next_speaker = None
        for agent in agents:
            if agent_speaks[agent.name] == min_speaks:
                next_speaker = agent
                break
        
        if next_speaker is None:
            break  # All agents have spoken their maximum times
            
        # Check if agent has used the tool
        if not agent_tool_usage[next_speaker.name]:
            # Force tool usage before speaking
            tool_message = Content(
                parts=[Part(text=f"""
                IMPORTANT: You MUST use the Tool to research your persona before speaking.
                This is a mandatory requirement. You cannot speak until you have used the Tool.
                
                Current conversation:
                {current_message}
                
                Use the Tool now to research your persona and then respond to the conversation.
                """)],
                role="user"
            )
            
            # Get tool usage response
            tool_response = None
            for event in runners[next_speaker.name].run(
                user_id="group_chat",
                session_id=f"session_{next_speaker.name}",
                new_message=tool_message
            ):
                if event.is_final_response() and event.content and event.content.parts:
                    tool_response = event.content.parts[0].text
                    break
            
            if tool_response:
                agent_tool_usage[next_speaker.name] = True
                # Broadcast tool usage
                if game_id and game_id in active_streams:
                    await active_streams[game_id].put(GameUpdate(
                        type="tool_usage",
                        data={
                            "agent": next_speaker.name,
                            "message": "Agent has completed required research using the Tool"
                        },
                        timestamp=get_gmt8_timestamp()
                    ).model_dump())
            else:
                # Skip this agent if they fail to use the tool
                continue
        
        # Create message content without the speaker prefix
        message_content = Content(
            parts=[Part(text=current_message)],
            role="user"
        )
        
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
            agent_speaks[current_speaker] += 1  # Increment speak count
            
            # Append into conversation logs.
            message_data = {"speaker": current_speaker, "message": current_message}
            complete_conversation.append(message_data)
            
            # Broadcast the message immediately if game_id is provided
            if game_id and game_id in active_streams:
                await active_streams[game_id].put(GameUpdate(
                    type="message",
                    data={"speaker": current_speaker, "message": current_message},
                    timestamp=get_gmt8_timestamp()
                ).model_dump())
            
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
            agent_speaks[current_speaker] += 1  # Count failed response as a speak
        
        # Small delay between messages
        await asyncio.sleep(2)
    
    return session_services, memory_services, runners, complete_conversation

@app.post("/games/{game_id}/round", response_model=GameState, tags=["Game Phases - Response will be broadcasted"])
async def play_round(game_id: str):
    """
    Play a single round of the game with real-time updates.
    """
    if game_id not in active_games:
        raise HTTPException(status_code=404, detail="Game not found")
        
    game = active_games[game_id]
    if game["state"]["status"] == "completed":
        raise HTTPException(status_code=400, detail="Game is already completed")
    
    try:
        # Clear previous round's searches
        clear_current_round_searches()
        
        # Update game status
        game["state"]["status"] = "in_progress"
        game["state"]["current_round"] += 1
        
        # Broadcast round start
        if game_id in active_streams:
            await active_streams[game_id].put(GameUpdate(
                type="status",
                data={"status": "in_progress", "round": game["state"]["current_round"]},
                timestamp=get_gmt8_timestamp()
            ).model_dump())
        
        # Phase 1: Communication Phase
        session_services, memory_services, runners, complete_conversation = await start_conversationFlow(
            game["agents"],
            game["state"]["keywords"],
            game_id=game_id  # Pass game_id to enable broadcasting
        )
        
        # Store services and conversation in game state
        game["session_services"] = session_services
        game["memory_services"] = memory_services
        game["runners"] = runners
        game["state"]["current_conversation"] = complete_conversation
        
        # Store conversation in history
        if "conversation_history" not in game["state"]:
            game["state"]["conversation_history"] = []
        
        # Get all searches from this round
        round_searches = get_current_round_searches()
        
        # Create round data with initial messages
        round_data = {
            "round": game["state"]["current_round"],
            "phase": "communication",
            "messages": complete_conversation,  # Add initial messages to the conversation
            "searches": round_searches
        }
        
        # Add to conversation history
        game["state"]["conversation_history"].append(round_data)
        
        # Phase 2: Analysis Phase - Now handled through API
        game["state"]["status"] = "waiting_for_analysis"
        game["state"]["current_phase"] = "analysis"
        
        # Store necessary data for analysis phase
        game["analysis_data"] = {
            "session_services": session_services,
            "memory_services": memory_services,
            "runners": runners,
            "complete_conversation": complete_conversation
        }
        
        # Broadcast analysis phase start
        if game_id in active_streams:
            await active_streams[game_id].put(GameUpdate(
                type="status",
                data={"status": "waiting_for_analysis", "phase": "analysis"},
                timestamp=get_gmt8_timestamp()
            ).model_dump())
        
        return game["state"]
    except Exception as e:
        game["state"]["status"] = "error"
        if game_id in active_streams:
            await active_streams[game_id].put(GameUpdate(
                type="status",
                data={"status": "error", "error": str(e)},
                timestamp=get_gmt8_timestamp()
            ).model_dump())
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/games/{game_id}", tags=["Game State"])
async def delete_game(game_id: str):
    """
    Delete a specific game instance.
    """
    if game_id not in active_games:
        raise HTTPException(status_code=404, detail="Game not found")
    
    del active_games[game_id]
    return {"message": "Game deleted successfully"}

@app.post("/games/{game_id}/analysis", response_model=AnalysisResponse, tags=["Game Phases - Response will be broadcasted"])
async def submit_analysis(game_id: str, user_input: UserInput):
    """
    Submit user input for agents to analyze during the analysis phase.
    """
    if game_id not in active_games:
        raise HTTPException(status_code=404, detail="Game not found")
    
    game = active_games[game_id]
    if game["state"]["status"] != "waiting_for_analysis":
        raise HTTPException(status_code=400, detail="Game is not in analysis phase")
    
    try:
        # Check for empty input or specific phrases indicating no response
        no_response_phrases = ["none", "no response", "nothing", "skip", "pass"]
        user_input_lower = user_input.input.lower().strip() if user_input.input else ""
        
        if (not user_input.input or 
            not user_input.input.strip() or 
            user_input_lower in no_response_phrases):
            
            # Store empty analysis in conversation history
            if "conversation_history" not in game["state"]:
                game["state"]["conversation_history"] = []
            
            game["state"]["conversation_history"].append({
                "round": game["state"]["current_round"],
                "phase": "analysis",
                "messages": [{"speaker": "User", "message": user_input.input or ""}],
                "analysis": {"user_input": user_input.input or "", "agent_responses": {}}
            })
            
            # Update game state to waiting for voting
            game["state"]["status"] = "waiting_for_voting"
            game["state"]["current_phase"] = "voting"
            
            # Broadcast analysis skip
            if game_id in active_streams:
                await active_streams[game_id].put(GameUpdate(
                    type="analysis",
                    data={"status": "skipped", "message": "No analysis requested"},
                    timestamp=get_gmt8_timestamp()
                ).model_dump())
            
            return {
                "agent_responses": {},
                "message": "No analysis requested. Skipping analysis phase. Call /games/{game_id}/vote to proceed with voting."
            }
        
        # Get the current conversation from game state
        conversation = game["state"].get("current_conversation", [])
        conversation_text = "\n".join([f"{item['speaker']}: {item['message']}" for item in conversation])
        
        # Get the analysis data from game state
        analysis_data = game.get("analysis_data", {})
        session_services = analysis_data.get("session_services", {})
        memory_services = analysis_data.get("memory_services", {})
        runners = analysis_data.get("runners", {})
        
        # Create message content for the user input
        message_content = Content(
            parts=[Part(text=f"""
        IMPORTANT: This is a question from the HUMAN USER, not from another agent.

        USER'S QUESTION: {user_input.input}

        Previous Conversation:
        {conversation_text}

        Please analyze the conversation and respond to the user's question above. Remember to:
        1. Use the required keywords naturally in your response
        2. Stay in character and maintain your persona
        3. Do not reveal your role
        4. Consider how your response might affect others' perception of you
        """)],
                    role="user"
                )
        
        # Get responses from each agent
        agent_responses = {}
        for agent in game["agents"]:
            if agent.name in runners:
                try:
                    # Get response from the agent
                    final_response = None
                    for event in runners[agent.name].run(
                        user_id="group_chat",
                        session_id=f"session_{agent.name}",
                        new_message=message_content
                    ):
                        if event.is_final_response() and event.content and event.content.parts:
                            final_response = event.content.parts[0].text
                            break
                    
                    if final_response:
                        agent_responses[agent.name] = final_response
                        
                        # Broadcast each agent's analysis response immediately
                        if game_id in active_streams:
                            await active_streams[game_id].put(GameUpdate(
                                type="analysis_response",
                                data={
                                    "agent": agent.name,
                                    "response": final_response,
                                    "user_input": user_input.input
                                },
                                timestamp=get_gmt8_timestamp()
                            ).model_dump())
                        
                        # Add the completed session to memory
                        completed_session = session_services[agent.name].get_session(
                            app_name="agent_conversation",
                            user_id="group_chat",
                            session_id=f"session_{agent.name}"
                        )
                        await memory_services[agent.name].add_session_to_memory(completed_session)
                except Exception as e:
                    print(f"Error getting response from agent {agent.name}: {str(e)}")
                    agent_responses[agent.name] = "I apologize, but I'm having trouble responding at the moment."
        
        # Store the responses in game state
        if "analyses" not in game["state"]:
            game["state"]["analyses"] = []
        
        analysis_data = {
            "user_input": user_input.input,
            "agent_responses": agent_responses
        }
        game["state"]["analyses"].append(analysis_data)
        
        # Store responses in conversation history
        game["state"]["conversation_history"].append({
            "round": game["state"]["current_round"],
            "phase": "analysis",
            "messages": [{"speaker": "User", "message": user_input.input}],
            "analysis": analysis_data
        })

        # Update game state to waiting for voting
        game["state"]["status"] = "waiting_for_voting"
        game["state"]["current_phase"] = "voting"
        
        # Broadcast analysis phase complete
        if game_id in active_streams:
            await active_streams[game_id].put(GameUpdate(
                type="status",
                data={"status": "waiting_for_voting", "phase": "voting"},
                timestamp=get_gmt8_timestamp()
            ).model_dump())
        
        return {
            "agent_responses": agent_responses,
            "message": "Analysis completed. Call /games/{game_id}/vote to proceed with voting."
        }
    except Exception as e:
        if game_id in active_streams:
            await active_streams[game_id].put(GameUpdate(
                type="status",
                data={"status": "error", "error": str(e)},
                timestamp=get_gmt8_timestamp()
            ).model_dump())
        raise HTTPException(status_code=500, detail=f"Error in analysis phase: {str(e)}")

@app.post("/games/{game_id}/vote", response_model=VotingResult, tags=["Game Phases - Response will be broadcasted"])
async def proceed_voting(game_id: str):
    """
    Proceed with the voting phase of the game.
    """
    if game_id not in active_games:
        raise HTTPException(status_code=404, detail="Game not found")
        
    game = active_games[game_id]
    if game["state"]["status"] != "waiting_for_voting":
        raise HTTPException(status_code=400, detail="Game is not in voting phase")
    
    try:
        # Get the latest analysis if it exists, otherwise use empty analysis
        latest_analysis = game["state"].get("analyses", [{}])[-1] if game["state"].get("analyses") else {
            "user_input": "",
            "agent_responses": {}
        }
        
        # Broadcast voting start
        if game_id in active_streams:
            await active_streams[game_id].put(GameUpdate(
                type="status",
                data={"status": "voting", "phase": "voting"},
                timestamp=get_gmt8_timestamp()
            ).model_dump())
        
        # Process voting session
        round_result = await voting_session(
            game["agents"],
            game["analysis_data"]["session_services"],
            game["analysis_data"]["memory_services"],
            game["analysis_data"]["runners"],
            game["analysis_data"]["complete_conversation"],
            [latest_analysis],
            game["werewolf"]
        )
        
        # Update game state based on voting result
        if round_result["action"] == "eliminate":
            eliminated_agent = round_result["eliminated_agent"].name
            game["agents"] = [agent for agent in game["agents"] if agent.name != eliminated_agent]
            game["state"]["eliminated_agents"].append(eliminated_agent)
            game["state"]["remaining_agents"] = [agent.name for agent in game["agents"]]
            
            # Check if werewolf was eliminated
            if eliminated_agent == game["werewolf"].name:
                game["state"]["status"] = "completed"
                game["state"]["result"] = "agents_win"
            elif len(game["agents"]) <= 2:
                game["state"]["status"] = "completed"
                game["state"]["result"] = "werewolf_win"
            else:
                game["state"]["status"] = "in_progress"
                game["state"]["current_phase"] = "communication"
        else:  # no_elimination
            if len(game["agents"]) <= 2:
                game["state"]["status"] = "completed"
                game["state"]["result"] = "werewolf_win"
            else:
                game["state"]["status"] = "in_progress"
                game["state"]["current_phase"] = "communication"
        
        # Add voting result to game history
        game["state"]["game_history"].append({
            "round": game["state"]["current_round"],
            "phase": "voting",
            "eliminated_agent": eliminated_agent if round_result["action"] == "eliminate" else None,
            "result": round_result["action"],
            "reason": round_result.get("reason", round_result.get("summary", "No reason provided")),
            "vote_details": round_result.get("vote_details", []),
            "vote_counts": round_result.get("vote_counts", {}),
            "abstain_count": round_result.get("abstain_count", 0)
        })
        
        # Broadcast voting results
        if game_id in active_streams:
            # First broadcast the action and eliminated agent
            await active_streams[game_id].put(GameUpdate(
                type="vote",
                data={
                    "action": round_result["action"],
                    "eliminated_agent": eliminated_agent if round_result["action"] == "eliminate" else None,
                    "game_status": game["state"]["status"]  # Include game status
                },
                timestamp=get_gmt8_timestamp()
            ).model_dump())
            
            # Add a delay before showing vote details
            await asyncio.sleep(2)
            
            # Broadcast each vote detail one by one
            for vote_detail in round_result.get("vote_details", []):
                await active_streams[game_id].put(GameUpdate(
                    type="vote_detail",
                    data={
                        "agent": vote_detail["agent"],
                        "vote": vote_detail["vote"],
                        "reasoning": vote_detail["reasoning"],
                        "confidence": vote_detail["confidence"],
                        "suspicion_level": vote_detail["suspicion_level"],
                        "game_status": game["state"]["status"]  # Include game status
                    },
                    timestamp=get_gmt8_timestamp()
                ).model_dump())
                # Add a delay between each vote detail
                await asyncio.sleep(2)
            
            # Add a delay before showing vote counts
            await asyncio.sleep(2)
            
            # Broadcast vote counts and abstain count
            await active_streams[game_id].put(GameUpdate(
                type="vote_summary",
                data={
                    "vote_counts": round_result.get("vote_counts", {}),
                    "abstain_count": round_result.get("abstain_count", 0),
                    "game_status": game["state"]["status"]  # Include game status
                },
                timestamp=get_gmt8_timestamp()
            ).model_dump())
            
            # Add a delay before showing game status update
            await asyncio.sleep(2)
            
            # Broadcast game status update
            await active_streams[game_id].put(GameUpdate(
                type="status",
                data={
                    "status": game["state"]["status"],
                    "phase": game["state"]["current_phase"],
                    "result": game["state"].get("result")
                },
                timestamp=get_gmt8_timestamp()
            ).model_dump())
        
        return {
            "eliminated_agent": eliminated_agent if round_result["action"] == "eliminate" else None,
            "round": game["state"]["current_round"],
            "result": round_result["action"],
            "reason": round_result.get("reason", round_result.get("summary", "No reason provided")),
            "remaining_agents": game["state"]["remaining_agents"],
            "eliminated_agents": game["state"]["eliminated_agents"],
            "vote_details": round_result.get("vote_details", []),
            "vote_counts": round_result.get("vote_counts", {}),
            "abstain_count": round_result.get("abstain_count", 0),
            "game_status": game["state"]["status"]  # Include game status in response
        }
    except Exception as e:
        game["state"]["status"] = "error"
        if game_id in active_streams:
            await active_streams[game_id].put(GameUpdate(
                type="status",
                data={"status": "error", "error": str(e)},
                timestamp=get_gmt8_timestamp()
            ).model_dump())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/games/{game_id}/conversation", response_model=List[Dict],
    summary="Get current conversation for the ROUND itself",
    description="Get the current conversation log for a game.",
    response_description="Returns the list of conversation messages", tags=["Game History"])
async def get_conversation(game_id: str):
    """
    Get the current conversation log for a game.
    
    Args:
        game_id (str): The ID of the game
        
    Returns:
        List[Dict]: List of conversation messages
    """
    if game_id not in active_games:
        raise HTTPException(status_code=404, detail="Game not found")
    
    return active_games[game_id]["state"].get("current_conversation", [])

@app.get("/games/{game_id}/conversation-history", response_model=List[ConversationHistory],
    summary="Get full conversation history for the whole entire game",
    description="Get the complete conversation history including all rounds, phases, and analyses.",
    response_description="Returns the complete conversation history", tags=["Game History"])
async def get_conversation_history(game_id: str):
    """
    Get the complete conversation history for the whole entire game.
    
    Args:
        game_id (str): The ID of the game
        
    Returns:
        List[ConversationHistory]: Complete conversation history
    """
    if game_id not in active_games:
        raise HTTPException(status_code=404, detail="Game not found")
    
    game = active_games[game_id]
    conversation_history = []
    
    # Get all rounds from conversation history
    for round_data in game["state"].get("conversation_history", []):
        # Create a new conversation history entry for each round and phase
        history_entry = {
            "round": round_data["round"],
            "phase": round_data["phase"],
            "messages": []
        }
        
        # Add messages if present
        if round_data.get("messages"):
            history_entry["messages"].extend(round_data["messages"])
        
        # Add search queries if present
        if round_data.get("searches"):
            for search in round_data["searches"]:
                history_entry["messages"].append({
                    "speaker": search["agent"],
                    "message": f"Search Query: {search['query']}",
                    "type": "search"
                })
                if search.get("result"):
                    history_entry["messages"].append({
                        "speaker": "System",
                        "message": f"Search Result: {json.dumps(search['result'])}",
                        "type": "search_result"
                    })
        
        # Add analysis if present
        if round_data.get("analysis"):
            analysis = round_data["analysis"]
            history_entry["messages"].append({
                "speaker": "User",
                "message": analysis.get("user_input", ""),
                "type": "analysis"
            })
            for agent_name, response in analysis.get("agent_responses", {}).items():
                history_entry["messages"].append({
                    "speaker": agent_name,
                    "message": response,
                    "type": "analysis_response"
                })
        

            voting_details = next(
                (vote for vote in game["state"].get("game_history", []) 
                 if vote.get("round") == round_data["round"] and vote.get("phase") == "voting"),
                None
            )
            
            if voting_details:
                # Add voting results
                history_entry["messages"].append({
                    "speaker": "System",
                    "message": f"Voting Results for Round {voting_details['round']}: Eliminated Agent: {voting_details.get('eliminated_agent', 'None')}, Result: {voting_details.get('result', 'No result')}",
                    "type": "voting_result"
                })
                
                # Add vote details
                for vote in voting_details.get("vote_details", []):
                    history_entry["messages"].append({
                        "speaker": vote["agent"],
                        "message": f"Voted for {vote['vote']} with confidence {vote['confidence']}",
                        "type": "vote"
                    })
                
                # Add vote counts
                history_entry["messages"].append({
                    "speaker": "System",
                    "message": f"Vote Counts: {json.dumps(voting_details.get('vote_counts', {}))}, Abstain Count: {voting_details.get('abstain_count', 0)}",
                    "type": "vote_summary"
                })
        
        conversation_history.append(history_entry)
    
    return conversation_history

@app.post("/games/{game_id}/analyze-conversation", response_model=ConversationAnalysis,
    summary="Analyze conversation history using Gemini 2.5 Pro",
    description="Analyze the conversation history to understand why each agent acted the way they did.", tags= ["Game Phases - Response will be broadcasted"])
async def analyze_conversation(game_id: str):
    """
    Analyze the conversation history using Gemini 2.5 Pro to understand agent behavior.
    
    Args:
        game_id (str): The ID of the game
        
    Returns:
        ConversationAnalysis: Analysis of agent behavior in the conversation
    """
    if game_id not in active_games:
        raise HTTPException(status_code=404, detail="Game not found")

    game = active_games[game_id]
    conversation_history = game["state"].get("conversation_history", [])
    
    if not conversation_history:
        raise HTTPException(status_code=400, detail="No conversation history available")
    
    agent_personas = game["state"]["agent_personas"]
    keywords_intialized = game["state"]["keywords"]
    
    # Get the latest round data
    latest_round = conversation_history[-1]
    
    # Prepare the conversation text - include all messages without splitting by rounds
    conversation_text = []
    for round_data in conversation_history:
        # Add all messages
        if round_data.get("messages"):
            conversation_text.extend([f"{item['speaker']}: {item['message']}" for item in round_data["messages"]])
        
        # Add search queries if present
        if round_data.get("searches"):
            conversation_text.append("\nSearch Queries:")
            for search in round_data["searches"]:
                conversation_text.append(f"{search['agent']} searched for: {search['query']}")
                if search.get("result"):
                    conversation_text.append(f"Search result: {json.dumps(search['result'], indent=2)}")
            conversation_text.append("")  # Add empty line for readability
        
        # Add analysis if present
        if round_data.get("analysis"):
            analysis = round_data["analysis"]
            conversation_text.append(f"User Analysis: {analysis.get('user_input', '')}")
            for agent_name, response in analysis.get("agent_responses", {}).items():
                conversation_text.append(f"{agent_name}'s Analysis: {response}")
        
        # Add voting details if present
        if round_data.get("phase") == "voting":
            voting_details = game["state"].get("game_history", [])
            for vote_round in voting_details:
                if vote_round["round"] == round_data["round"]:
                    conversation_text.append(f"Voting Results for Round {vote_round['round']}:")
                    conversation_text.append(f"Eliminated Agent: {vote_round.get('eliminated_agent', 'None')}")
                    conversation_text.append(f"Result: {vote_round.get('result', 'No result')}")
                    conversation_text.append(f"Reason: {vote_round.get('reason', 'No reason provided')}")
                    conversation_text.append("Vote Details:")
                    for vote in vote_round.get("vote_details", []):
                        conversation_text.append(f"- {vote['agent']} voted for {vote['vote']} with confidence {vote['confidence']}")
                    conversation_text.append(f"Vote Counts: {json.dumps(vote_round.get('vote_counts', {}), indent=2)}")
                    conversation_text.append(f"Abstain Count: {vote_round.get('abstain_count', 0)}")
    
    conversation_text = "\n".join(conversation_text)
    
    # Get voting history
    voting_history = []
    for round_data in game["state"].get("game_history", []):
        if round_data.get("phase") == "voting":
            voting_history.append({
                "round": round_data["round"],
                "eliminated_agent": round_data.get("eliminated_agent"),
                "result": round_data["result"],
                "reason": round_data.get("reason", "No reason provided"),
                "vote_details": round_data.get("vote_details", []),
                "vote_counts": round_data.get("vote_counts", {}),
                "abstain_count": round_data.get("abstain_count", 0)
            })
    
    # Create the analysis prompt for Gemini 2.5 Pro
    analysis_prompt = f"""
    You are a highly advanced behavioral analyst tasked with conducting a comprehensive, deeply layered analysis of a multi-agent social deduction game. This analysis must be thorough, nuanced, and extensive — drawing on social psychology, communication theory, and behavioral profiling. Leave no detail unexplored.

    This is a high-stakes environment involving deception, influence, and group strategy. Your goal is to dissect each agent's behavior with **maximum depth**, **clear evidence**, and **detailed pattern recognition**, producing an analysis that is **long, multi-paragraph**, and **rich with insight** for every agent.

    Contextual Information:
    - **Keywords used during the game**:  
    {keywords_intialized}

    - **Agent personas (declared or inferred roles/personalities)**:  
    {agent_personas}

    - **Complete conversation transcript**:  
    {conversation_text}

    - **Voting history summary**:  
    {json.dumps(voting_history, indent=2)}

    ---

    For each agent involved in the conversation, generate a **long-form analysis** that includes:

    1. **Communication Patterns**:
    - How frequently they contributed and at what key moments
    - Message length and cognitive complexity (e.g. concise, verbose, hedging)
    - Tone, style, and vocabulary — was it assertive, passive, evasive, performative?
    - Relationship patterns (who they responded to, who they ignored)
    - How and when they deployed the game's keywords for strategic advantage

    2. **Behavioral Traits**:
    - Dominance/submissiveness, leadership attempts, or follower tendencies
    - Clarity and consistency in decision-making
    - Emotional tone (flat, reactive, empathetic, guarded)
    - Adaptability in the face of changing alliances or social pressure

    3. **Strategic Elements**:
    - Their strategy for revealing or concealing information
    - Questioning style (manipulative, investigative, disarming)
    - Consistency or evolution of their claims over time
    - Control over the topic or redirection tactics
    - Use of keywords as trust-building or manipulation tools

    4. **Social Interaction**:
    - Role in the group dynamic: influencer, scapegoat, observer, etc.
    - Persuasive tactics and attempts to shape group consensus
    - Responses to confrontation, praise, or suspicion
    - Conflict style: avoidance, escalation, diplomacy
    - Whether their communication matched their voting behavior

    ---

    **Key Instructions:**
    - Your analysis **must be long, granular, and heavily evidence-based**.
    - Write **multiple detailed paragraphs per agent**, citing specific behavior, language, and interactions.
    - Do **not** summarize or generalize — go deep into behavioral micro-patterns and possible motives.
    - Include **key actions** and **supporting communication patterns** per agent.
    - Assign a confidence score indicating how certain your observations are, based on behavioral evidence.

    ---

    **Return the analysis as a JSON object with this exact structure**:

    {{
    "agent_analyses": [
        {{
        "agent_name": "agent_name",
        "behavior_analysis": "Detailed, long-form analysis of the agent's behavior including communication patterns, behavioral traits, strategic elements, and social interaction",
        "key_actions": ["action1", "action2", ...],
        "suspicious_patterns": ["pattern1", "pattern2", ...],
        "confidence_score": 0.85
        }},
    ],
    "overall_summary": "A multi-paragraph, in-depth summary that outlines the overarching dynamics, patterns of communication, alliances, deception tactics, and key psychological behaviors observed in the entire group."
    }}
    """
    
    try:
        # Initialize Gemini client
        client = genai.Client(http_options=HttpOptions(api_version="v1"))
        
        # Get analysis from Gemini
        response = client.models.generate_content(
            model="gemini-2.0-flash-001",  # Using a more stable model
            contents=analysis_prompt
        )
        
        # Get the response text
        response_text = response.text
        
        # Debug print
        print("Raw response from Gemini:", response_text)
        
        try:
            # Try to parse the response as JSON
            analysis_data = json.loads(response_text)
        except json.JSONDecodeError as json_err:
            # If JSON parsing fails, try to extract JSON from the text
            # Look for JSON-like structure in the response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    analysis_data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    # If still can't parse, try to clean the response
                    cleaned_text = response_text.replace('```json', '').replace('```', '').strip()
                    try:
                        analysis_data = json.loads(cleaned_text)
                    except json.JSONDecodeError:
                        raise HTTPException(
                            status_code=500,
                            detail=f"Could not parse model response as JSON. Raw response: {response_text[:200]}..."
                        )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Model response is not in JSON format. Raw response: {response_text[:200]}..."
                )
        
        # Validate the response structure
        if not isinstance(analysis_data, dict):
            raise HTTPException(
                status_code=500,
                detail="Model response is not a valid JSON object"
            )
            
        if "agent_analyses" not in analysis_data or "overall_summary" not in analysis_data:
            raise HTTPException(
                status_code=500,
                detail="Model response missing required fields: agent_analyses or overall_summary"
            )
        
        # Create the response object
        return analysis_data
        
    except Exception as e:
        print(f"Error in analyze_conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing conversation: {str(e)}")

@app.get("/games/{game_id}/get-agent-personas", response_model=AgentPersonas,
    summary="Get game information including agent personas and keywords",
    description="Retrieves the agent personas and keywords for a specific game.", tags=["Game History"])
async def get_game_info(game_id: str):
    """
    Get the agent personas and keywords for a specific game.
    
    Args:
        game_id (str): The ID of the game
        
    Returns:
        GameInfo: Information about agent personas and keywords in the game
    """
    if game_id not in active_games:
        raise HTTPException(status_code=404, detail="Game not found")
    
    game = active_games[game_id]
    
    return AgentPersonas(
        agent_personas=game["state"]["agent_personas"]
    )

@app.get("/games/{game_id}/get-keywords", response_model=SelectedKeywords,
         summary="Retrieves the keywords initialized for the specific game id",
         description="Returns the keywords initialized for the whole game", tags=["Game History"]
         )
async def get_keywords(game_id:str):

    if game_id not in active_games:
        raise HTTPException(status_code=404, detail="Game not found")
    
    game = active_games[game_id]

    return SelectedKeywords(
        keywords=game["state"]["keywords"]
    )


"""
@app.get("/games/{game_id}/search-logs-history", response_model=List[RoundSearchLogs],
    summary="Get web search logs",
    description="Get the logs of web searches performed by agents during the game, organized by rounds.",
    response_description="Returns the list of web search logs organized by rounds")
async def get_search_logs_endpoint(game_id: str):

    if game_id not in active_games:
        raise HTTPException(status_code=404, detail="Game not found")
    
    game = active_games[game_id]
    round_logs = []
    
    # Get all rounds from conversation history
    for round_data in game["state"].get("conversation_history", []):
        if round_data.get("phase") == "communication" and "searches" in round_data:
            round_logs.append({
                "round": round_data["round"],
                "searches": round_data["searches"]
            })
    
    return round_logs
"""

"""
@app.delete("/games/{game_id}/search-logs-history",
    summary="Clear web search logs",
    description="Clear all web search logs for the game.")
async def clear_search_logs_endpoint(game_id: str):
    if game_id not in active_games:
        raise HTTPException(status_code=404, detail="Game not found")
    
    clear_search_logs()
    return {"message": "Search logs cleared successfully"}
"""

"""
@app.get("/games/{game_id}/current-round", response_model=RoundData)
async def get_current_round(game_id: str):
    if game_id not in active_games:
        raise HTTPException(status_code=404, detail="Game not found")
    
    game = active_games[game_id]
    conversation_history = game["state"].get("conversation_history", [])
    round_searches = get_current_round_searches()
    
    if not conversation_history:
        return RoundData(
            round=game["state"]["current_round"],
            phase=game["state"].get("current_phase", "initialization"),
            messages=[],
            searches=round_searches,
            status=game["state"]["status"]
        )
    
    # Get the latest round data
    latest_round = conversation_history[-1]
    
    return RoundData(
        round=latest_round["round"],
        phase=latest_round["phase"],
        messages=latest_round.get("messages", []),
        searches=latest_round.get("searches", round_searches),
        status=game["state"]["status"]
    )
"""
