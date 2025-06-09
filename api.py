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
from datetime import datetime, timedelta
import uvicorn
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
    duration_minutes: float = 5.0
    model: str = "gemini-2.0-flash"
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
                "input": "What do you think about the conversation so far?"
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
    round: int
    phase: str
    agent_analyses: List[AgentAnalysis]
    overall_summary: str

class AgentPersonas(BaseModel):
    agent_personas: List[Dict]

class SelectedKeywords(BaseModel):
    keywords: List[str]

# Store active games
active_games = {}

# Store active streams for each game
active_streams = {}

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
                f"\n\nIMPORTANT: You can ONLY use these keywords in your responses: {', '.join(keywords)}. "
                "You must incorporate as many keywords as possible in each response. "
                "Be creative in how you use these keywords while maintaining your persona, but do not mention your persona or role."
                "Do not reveal your persona or role to the other agents."
                "Identify your persona and use the tool to search for information about your persona and its strategies."
                "You are not allowed to talk to the other agents until you have used the tool for research."
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
            "timestamp": datetime.now().isoformat()
        }
        yield f"data: {json.dumps(initial_message)}\n\n"
        
        # Send a test message
        test_message = {
            "type": "status",
            "data": {"status": "test", "message": "Stream is working"},
            "timestamp": datetime.now().isoformat()
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
                    "timestamp": datetime.now().isoformat()
                }
                yield f"data: {json.dumps(keep_alive)}\n\n"
                
    except Exception as e:
        print(f"Error in game stream: {str(e)}")
        error_message = {
            "type": "error",
            "data": {"error": str(e)},
            "timestamp": datetime.now().isoformat()
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

async def call_conversationFlow(agents, keywords, duration_minutes, game_id=None):
    """
    Run a conversation between agents for a specified duration using only allowed keywords.
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
            message_data = {"speaker": current_speaker, "message": current_message}
            complete_conversation.append(message_data)
            
            # Broadcast the message immediately if game_id is provided
            if game_id and game_id in active_streams:
                await active_streams[game_id].put(GameUpdate(
                    type="message",
                    data={"speaker": current_speaker, "message": current_message},
                    timestamp=datetime.now().isoformat()
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
                timestamp=datetime.now().isoformat()
            ).model_dump())
        
        # Phase 1: Communication Phase
        session_services, memory_services, runners, complete_conversation = await call_conversationFlow(
            game["agents"],
            game["state"]["keywords"],
            duration_minutes=1.0,
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
        
        # Create round data
        round_data = {
            "round": game["state"]["current_round"],
            "phase": "communication",
            "messages": complete_conversation,
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
                timestamp=datetime.now().isoformat()
            ).model_dump())
        
        return game["state"]
    except Exception as e:
        game["state"]["status"] = "error"
        if game_id in active_streams:
            await active_streams[game_id].put(GameUpdate(
                type="status",
                data={"status": "error", "error": str(e)},
                timestamp=datetime.now().isoformat()
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
                "messages": [{"speaker": "user", "message": user_input.input or ""}],
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
                    timestamp=datetime.now().isoformat()
                ).model_dump())
            
            return {
                "agent_responses": {},
                "message": "No analysis requested. Skipping analysis phase. Call /games/{game_id}/vote to proceed with voting."
            }
        
        # Get the current conversation from game state
        conversation = game["state"].get("current_conversation", [])
        conversation_text = "\n".join([f"{item['speaker']}: {item['message']}" for item in conversation])
        
        # Create the analysis prompt for Gemini 2.5 Pro
        analysis_prompt = f"""
        Analyze the following conversation from a social deduction game and provide a detailed behavioral analysis of each agent.
        Focus on observable behaviors and communication patterns.
        
        Conversation:
        {conversation_text}
        
        For each agent in the conversation, analyze their behavior in detail:
        1. Communication Patterns:
           - Message frequency and timing
           - Response length and complexity
           - Language style and vocabulary usage
           - Interaction patterns with other agents
        
        2. Behavioral Traits:
           - Leadership tendencies
           - Decision-making approach
           - Emotional expression
           - Adaptability to group dynamics
        
        3. Strategic Elements:
           - Information sharing patterns
           - Question asking behavior
           - Response consistency
           - Topic control and direction
        
        4. Social Interaction:
           - Group role and positioning
           - Influence attempts
           - Response to others' statements
           - Conflict handling approach
        
        Format your response as a JSON object with the following structure:
        {{
            "agent_analyses": [
                {{
                    "agent_name": "agent_name",
                    "behavior_analysis": {{
                        "communication_patterns": "detailed analysis of communication style and patterns",
                        "behavioral_traits": "analysis of consistent behavioral characteristics",
                        "strategic_elements": "analysis of strategic approach and consistency",
                        "social_interaction": "analysis of social behavior and group dynamics"
                    }},
                    "key_actions": ["action1", "action2", ...],
                    "confidence_score": 0.85
                }},
                ...
            ],
            "overall_summary": {{
                "group_dynamics": "Analysis of how the group interacts as a whole",
                "communication_patterns": "Overview of communication styles across agents",
                "behavioral_trends": "Common behavioral patterns observed",
                "strategic_landscape": "Overview of the strategic situation"
            }}
        }}
        """
        
        # Initialize Gemini client
        client = genai.Client(http_options=HttpOptions(api_version="v1"))
        
        # Get analysis from Gemini
        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-05-20",
            contents=analysis_prompt
        )
        
        # Parse the response
        analysis_data = json.loads(response.text)
        
        # Store the analysis in game state
        if "analyses" not in game["state"]:
            game["state"]["analyses"] = []
        
        analysis_data = {
            "user_input": user_input.input,
            "agent_responses": analysis_data["agent_analyses"]
        }
        game["state"]["analyses"].append(analysis_data)
        
        # Store analysis in conversation history
        game["state"]["conversation_history"].append({
            "round": game["state"]["current_round"],
            "phase": "analysis",
            "messages": [{"speaker": "user", "message": user_input.input}],
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
                timestamp=datetime.now().isoformat()
            ).model_dump())
        
        return {
            "agent_responses": analysis_data["agent_analyses"],
            "message": "Analysis completed. Call /games/{game_id}/vote to proceed with voting."
        }
    except Exception as e:
        if game_id in active_streams:
            await active_streams[game_id].put(GameUpdate(
                type="status",
                data={"status": "error", "error": str(e)},
                timestamp=datetime.now().isoformat()
            ).model_dump())
        raise HTTPException(status_code=500, detail=str(e))

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
                timestamp=datetime.now().isoformat()
            ).model_dump())
        
        # Process voting session
        round_result = await voting_session(
            game["agents"],
            game["analysis_data"]["session_services"],
            game["analysis_data"]["memory_services"],
            game["analysis_data"]["runners"],
            game["analysis_data"]["complete_conversation"],
            [latest_analysis],
            game["werewolf"],
            game["config"].duration_minutes
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
                timestamp=datetime.now().isoformat()
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
                    timestamp=datetime.now().isoformat()
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
                timestamp=datetime.now().isoformat()
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
                timestamp=datetime.now().isoformat()
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
                timestamp=datetime.now().isoformat()
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
    return game["state"].get("conversation_history", [])

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

    #include persona and keyword into here and we are done
    
    game = active_games[game_id]
    conversation_history = game["state"].get("conversation_history", [])
    
    if not conversation_history:
        raise HTTPException(status_code=400, detail="No conversation history available")
    
    # Get the latest round's conversation
    latest_round = conversation_history[-1]
    
    # Prepare the conversation text
    conversation_text = "\n".join([f"{item['speaker']}: {item['message']}" for item in latest_round.get("messages", [])])
    agent_personas = game["state"]["agent_personas"]
    keywords_intialized = game["state"]["keywords"]
    
    # Create the analysis prompt for Gemini 2.5 Pro
    analysis_prompt = f"""
    Analyze the following conversation from a social deduction game and provide a detailed behavioral analysis of each agent.
    Focus on observable behaviors and communication patterns.

    Keywords that were chosen for this game:
    {keywords_intialized}

    Agent's Personas:
    {agent_personas}
    
    Conversation:
    {conversation_text}
    
    For each agent in the conversation, analyze their behavior in detail:
    1. Communication Patterns:
       - Message frequency and timing
       - Response length and complexity
       - Language style and vocabulary usage
       - Interaction patterns with other agents
    
    2. Behavioral Traits:
       - Leadership tendencies
       - Decision-making approach
       - Emotional expression
       - Adaptability to group dynamics
    
    3. Strategic Elements:
       - Information sharing patterns
       - Question asking behavior
       - Response consistency
       - Topic control and direction
    
    4. Social Interaction:
       - Group role and positioning
       - Influence attempts
       - Response to others' statements
       - Conflict handling approach
    
    Format your response as a JSON object with the following structure:
    {{
        "agent_analyses": [
            {{
                "agent_name": "agent_name",
                "behavior_analysis": {{
                    "communication_patterns": "detailed analysis of communication style and patterns",
                    "behavioral_traits": "analysis of consistent behavioral characteristics",
                    "strategic_elements": "analysis of strategic approach and consistency",
                    "social_interaction": "analysis of social behavior and group dynamics"
                }},
                "key_actions": ["action1", "action2", ...],
                "confidence_score": 0.85
            }},
        ],
        "overall_summary": {{
            "group_dynamics": "Analysis of how the group interacts as a whole",
            "communication_patterns": "Overview of communication styles across agents",
            "behavioral_trends": "Common behavioral patterns observed",
            "strategic_landscape": "Overview of the strategic situation"
        }}
    }}
    """
    
    try:
        # Initialize Gemini client
        client = genai.Client(http_options=HttpOptions(api_version="v1"))
        
        # Get analysis from Gemini
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=analysis_prompt
        )
        
        # Parse the response
        analysis_data = json.loads(response.text)
        
        # Create the response object
        return ConversationAnalysis(
            round=latest_round["round"],
            phase=latest_round["phase"],
            agent_analyses=[
                AgentAnalysis(**analysis) for analysis in analysis_data["agent_analyses"]
            ],
            overall_summary=analysis_data["overall_summary"]
        )
        
    except Exception as e:
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
