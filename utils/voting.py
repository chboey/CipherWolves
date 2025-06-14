import asyncio
import re
import math
from google.genai.types import Content, Part

async def conduct_voting_round(remaining_agents, session_services, memory_services, runners,
                              complete_conversation, user_analyses, game_history, round_number):
    """
    Conduct a single voting round.
    
    Returns:
        dict: Round result with action and details
    """
    # Prepare context for this round
    conversation_text = "\n".join([f"{item['speaker']}: {item['message']}" for item in complete_conversation])
    
    # Include user analysis context
    analysis_context = ""
    if user_analyses:
        analysis_context = "\n\nHUMAN USER'S ANALYSIS PHASE:\n"
        for analysis in user_analyses:
            analysis_context += f"\nHUMAN USER asked: {analysis['user_input']}\n"
            for agent_name, response in analysis['agent_responses'].items():
                if any(agent.name == agent_name for agent in remaining_agents):  # Only include remaining agents
                    analysis_context += f"{agent_name} responded to the HUMAN USER: {response[:200]}...\n"
    
    # Include game history context
    history_context = ""
    if game_history:
        history_context = "\n\nPrevious Voting Rounds:\n"
        for i, round_data in enumerate(game_history, 1):
            history_context += f"Round {i}: {round_data['summary']}\n"
    
    # Create agent names list for voting options
    agent_names = [agent.name for agent in remaining_agents]
    
    voting_prompt = f"""
    You are participating in the Round of the werewolf elimination game.
    
    CURRENT SITUATION:
    - Remaining Players: {', '.join(agent_names)}
    - Round: {round_number}
    
    ORIGINAL CONVERSATION:
    {conversation_text}
    
    {analysis_context}
    
    {history_context}
    
    VOTING INSTRUCTIONS:
    Based on all available information, vote to ELIMINATE one player other than yourself that you believe is the werewolf.
    You have been provided with:
    1. The original conversation between agents
    2. Questions asked by the HUMAN USER during the analysis phase and how each agent responded
    3. Previous voting history
    
    When you see your name, refer to yourself as I. Example: As I have said earlier.
    
    Analyze these aspects:
    1. Keyword usage patterns and consistency
    2. Response patterns and behavioral analysis
    3. Previous voting patterns and defensive behavior
    4. Overall suspicion level based on all interactions
    5. How agents responded to the HUMAN USER's analysis questions - were they evasive, defensive, or cooperative?
    
    VOTING OPTIONS:
    - Vote to ELIMINATE a specific agent that is not yourself.
    - ABSTAIN: If you don't have enough evidence (but be careful - abstaining helps the werewolf!)
    
    IMPORTANT GAME RULES:
    - If MAJORITY votes for the same agent → That agent is ELIMINATED
    - If MAJORITY abstains → No elimination this round
    - If there's a TIE → No elimination this round
    - Game continues until werewolf is found or only 2 players remain
    
    Please provide your vote in this EXACT format:
    VOTE: [AGENT_NAME or ABSTAIN]
    REASONING: [Your detailed reasoning for the vote]
    CONFIDENCE: [LOW/MEDIUM/HIGH]
    SUSPICION_LEVEL: [How suspicious you are of your target on a scale 1-10]
    
    Example:
    VOTE: Agent_B
    REASONING: Agent_B's responses have been inconsistent and their keyword usage feels forced. They also gave evasive answers to the HUMAN USER's analysis questions, which is suspicious behavior.
    CONFIDENCE: HIGH
    SUSPICION_LEVEL: 8
    """
    
    # Get votes sequentially from remaining agents
    votes = []
    for agent in remaining_agents:
        try:
            vote = await get_agent_vote(
                agent, 
                runners[agent.name], 
                session_services[agent.name], 
                memory_services[agent.name], 
                voting_prompt
            )
            votes.append(vote)
        except Exception as e:
            votes.append((agent.name, {"vote": "ABSTAIN", "reasoning": f"Error: {str(e)}", "confidence": "NONE"}))
        
        # Add a small delay between votes to make it more natural
        await asyncio.sleep(2)
    
    # Process voting results
    return await process_round_results(votes, remaining_agents, round_number)


async def get_agent_vote(agent, runner, session_service, memory_service, voting_prompt):
    """
    Get vote from a single agent.
    
    Returns:
        tuple: (agent_name, vote_data)
    """
    # Create voting content
    voting_content = Content(
        parts=[Part(text=voting_prompt)],
        role="user"
    )
    voting_session_id = f"session_{agent.name}"
    
    try:
        session = session_service.get_session(
            app_name="agent_conversation",
            user_id="group_chat",
            session_id=voting_session_id
        )
    
        if not session:
            # Create new session if it doesn't exist
            session = session_service.create_session(
                app_name="agent_conversation",
                user_id="group_chat",
                session_id=voting_session_id,
                state={"status": "initialized"}
            )
            
        if not session:
            raise ValueError(f"Failed to create/get session: {voting_session_id}")
        
        
        # Get vote from agent
        vote_response = None
        for event in runner.run(
            user_id="group_chat",
            session_id=voting_session_id,
            new_message=voting_content
        ):
            if event.is_final_response() and event.content and event.content.parts:
                vote_response = event.content.parts[0].text
                break
        
        if not vote_response:
            vote_response = "VOTE: ABSTAIN\nREASONING: No response provided\nCONFIDENCE: NONE"
        
        # Parse the vote response
        vote_data = parse_vote_response(vote_response)
        
        # Add the completed session to memory
        completed_session = session_service.get_session(
            app_name="agent_conversation",
            user_id="group_chat",
            session_id=voting_session_id
        )
        await memory_service.add_session_to_memory(completed_session)
    
        return agent.name, vote_data
    
    except Exception as e:
        # print(f"Error in agent voting for {agent.name}: {str(e)}")
        return agent.name, f"Error during voting phase: {str(e)}"


def parse_vote_response(response):
    """
    Parse the agent's vote response to extract vote, reasoning, confidence, and suspicion level.
    
    Returns:
        dict: Parsed vote data
    """
    vote_data = {
        "vote": "ABSTAIN",
        "reasoning": "No reasoning provided",
        "confidence": "NONE",
        "suspicion_level": 0,
        "raw_response": response
    }
    
    # Extract vote
    vote_match = re.search(r'VOTE:\s*([^\n]+)', response, re.IGNORECASE)
    if vote_match:
        vote_data["vote"] = vote_match.group(1).strip()
    
    # Extract reasoning
    reasoning_match = re.search(r'REASONING:\s*([^\n]+(?:\n(?!CONFIDENCE:|SUSPICION_LEVEL:)[^\n]+)*)', response, re.IGNORECASE)
    if reasoning_match:
        vote_data["reasoning"] = reasoning_match.group(1).strip()
    
    # Extract confidence
    confidence_match = re.search(r'CONFIDENCE:\s*([^\n]+)', response, re.IGNORECASE)
    if confidence_match:
        vote_data["confidence"] = confidence_match.group(1).strip().upper()
    
    # Extract suspicion level
    suspicion_match = re.search(r'SUSPICION_LEVEL:\s*([^\n]+)', response, re.IGNORECASE)
    if suspicion_match:
        try:
            vote_data["suspicion_level"] = int(re.search(r'\d+', suspicion_match.group(1)).group())
        except:
            vote_data["suspicion_level"] = 0
    
    return vote_data

async def process_round_results(votes, remaining_agents, round_number):
    """
    Process the voting results and determine the round outcome.
    
    Returns:
        dict: Round result data
    """
    
    # Count votes
    vote_counts = {}
    abstain_count = 0
    total_votes = len(votes)
    majority_threshold = math.ceil(total_votes / 2)
    
    # Store detailed vote information
    vote_details = []
    
    # Display individual votes and count them
    for agent_name, vote_data in votes:
        
        # Add vote details to the list
        vote_details.append({
            "agent": agent_name,
            "vote": vote_data["vote"],
            "reasoning": vote_data["reasoning"],
            "confidence": vote_data["confidence"],
            "suspicion_level": vote_data["suspicion_level"]
        })
        
        # Count the vote
        vote = vote_data['vote'].upper().strip()
        if vote == "ABSTAIN":
            abstain_count += 1
        else:
            # Try to match vote to actual agent names
            matched_agent = None
            for agent in remaining_agents:
                if agent.name.upper() == vote.upper():
                    matched_agent = agent.name
                    break
            
            if matched_agent:
                vote_counts[matched_agent] = vote_counts.get(matched_agent, 0) + 1
            else:
                # Invalid vote counts as abstain
                abstain_count += 1
    
    # Wait for all votes to be processed before showing outcome
    await asyncio.sleep(1)  # Small delay to ensure all vote displays are complete
    
    
    # Check for majority abstention
    if abstain_count >= majority_threshold:
        # print("🤝 MAJORITY ABSTAINED - No elimination this round")
        return {
            "action": "no_elimination",
            "reason": "Majority abstained",
            "vote_counts": vote_counts,
            "abstain_count": abstain_count,
            "summary": f"Majority abstained ({abstain_count}/{total_votes})",
            "vote_details": vote_details
        }
    
    # Check for majority vote on a specific agent
    majority_votes = [(name, count) for name, count in vote_counts.items() if count >= majority_threshold]
    
    if len(majority_votes) == 1:
        eliminated_name = majority_votes[0][0]
        vote_count = majority_votes[0][1]
        
        # Find the agent object
        eliminated_agent = next(agent for agent in remaining_agents if agent.name == eliminated_name)
        
        # print(f"⚖️ MAJORITY DECISION - {eliminated_name} is ELIMINATED ({vote_count} votes)")
        
        return {
            "action": "eliminate",
            "eliminated_agent": eliminated_agent,
            "vote_count": vote_count,
            "vote_counts": vote_counts,
            "abstain_count": abstain_count,
            "summary": f"{eliminated_name} eliminated with {vote_count}/{total_votes} votes",
            "vote_details": vote_details
        }
    
    elif len(majority_votes) > 1:
        # Multiple agents got majority (shouldn't happen with proper majority calculation)
        tied_agents = [name for name, count in majority_votes]
        return {
            "action": "no_elimination",
            "reason": "Tied majority votes",
            "tied_agents": tied_agents,
            "vote_counts": vote_counts,
            "abstain_count": abstain_count,
            "summary": f"Tied majority votes between {', '.join(tied_agents)}",
            "vote_details": vote_details
        }
    
    else:
        # No majority achieved
        if vote_counts:
            highest_votes = max(vote_counts.values())
            tied_candidates = [name for name, count in vote_counts.items() if count == highest_votes]
            
            if len(tied_candidates) == 1:
                reason = f"No majority achieved - {tied_candidates[0]} had {highest_votes}/{total_votes} votes"
            else:
                reason = f"Tie between {', '.join(tied_candidates)} with {highest_votes} votes each"
        else:
            # print("❌ NO VOTES CAST - All abstained or invalid votes")
            reason = "No valid votes cast"
        
        return {
            "action": "no_elimination",
            "reason": reason,
            "vote_counts": vote_counts,
            "abstain_count": abstain_count,
            "summary": reason,
            "vote_details": vote_details
        }
