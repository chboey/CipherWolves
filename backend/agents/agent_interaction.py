from google.genai.types import Content, Part

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
    

    user_analyses = []
    
    # Process each agent's analysis sequentially
    agent_responses = {}
    for agent in agents:
        final_response = None
        async for agent_name, response in get_agent_analysis(
            agent, 
            runners[agent.name], 
            session_services[agent.name], 
            memory_services[agent.name], 
            conversation_text
        ):
            final_response = response
            
        if final_response:
            agent_responses[agent_name] = final_response
            
    user_analyses.append({
        "user_input": "Analysis phase",
        "agent_responses": agent_responses,
    })
    
    return user_analyses

async def get_agent_analysis(agent, runner, session_service, memory_service, analysis_prompt):
    """
    Get analysis from a main agent and its metric sub-agents.
    
    Args:
        agent: The main agent to get analysis from
        runner: The runner for the main agent
        session_service: The session service for the main agent
        memory_service: The memory service for the main agent
        analysis_prompt: The prompt for analysis
    
    Yields:
        tuple: (agent_name, analysis_response) for intermediate results
    """
    # Get metric sub-agent analyses if available
    metric_analyses = {}
    if hasattr(agent, 'sub_agents') and agent.sub_agents:
        for sub_agent in agent.sub_agents:
            # Create analysis content for sub-agent
            sub_analysis_content = Content(
                parts=[Part(text=f"""
                Analyze the following conversation from your perspective:
                
                {analysis_prompt}
                
                Provide your analysis in this EXACT format:
                trust_level: <float between 0 and 1>
                suspicion_level: <float between 0 and 1>
                target: <name of the agent you're analyzing>
                reason: <detailed explanation of your analysis>
                """)],
                role="user"
            )
            
            # Create a new session for this sub-agent analysis
            sub_analysis_session_id = f"session_{sub_agent.name}"
            
            try:
                # Get or create session for sub-agent
                sub_session = session_service.get_session(
                    app_name="agent_conversation",
                    user_id="group_chat",
                    session_id=sub_analysis_session_id
                )
                
                if not sub_session:
                    sub_session = session_service.create_session(
                        app_name="agent_conversation",
                        user_id="group_chat",
                        session_id=sub_analysis_session_id,
                        state={"status": "initialized"}
                    )
                
                if not sub_session:
                    raise ValueError(f"Failed to create/get session: {sub_analysis_session_id}")
                
                # Get analysis from sub-agent
                sub_final_response = None
                for event in runner.run(
                    user_id="group_chat",
                    session_id=sub_analysis_session_id,
                    new_message=sub_analysis_content
                ):
                    if event.is_final_response() and event.content and event.content.parts:
                        sub_final_response = event.content.parts[0].text
                        break
                
                if sub_final_response:
                    # Parse the structured response
                    try:
                        lines = sub_final_response.strip().split('\n')
                        metrics = {}
                        for line in lines:
                            if ':' in line:
                                key, value = line.split(':', 1)
                                metrics[key.strip()] = value.strip()
                        
                        # Store the parsed metrics
                        metric_analyses[sub_agent.name] = {
                            'trust_level': float(metrics.get('trust_level', 0)),
                            'suspicion_level': float(metrics.get('suspicion_level', 0)),
                            'target': metrics.get('target', 'unknown'),
                            'reason': metrics.get('reason', 'No reason provided')
                        }
                    except Exception as e:
                        metric_analyses[sub_agent.name] = {
                            'error': f"Failed to parse metrics: {str(e)}",
                            'raw_response': sub_final_response
                        }
                
                # Add the completed session to memory
                completed_sub_session = session_service.get_session(
                    app_name="agent_conversation",
                    user_id="group_chat",
                    session_id=sub_analysis_session_id
                )
                if completed_sub_session:
                    await memory_service.add_session_to_memory(completed_sub_session)
                    
            except Exception as e:
                metric_analyses[sub_agent.name] = {
                    'error': f"Error in sub-agent analysis: {str(e)}"
                }
    
    # Add metric analyses to the main prompt
    if metric_analyses:
        analysis_prompt += "\n\nMetric Sub-Agent Analyses:\n"
        for sub_agent_name, analysis in metric_analyses.items():
            if 'error' in analysis:
                analysis_prompt += f"\n{sub_agent_name} error: {analysis['error']}\n"
            else:
                analysis_prompt += f"""
                {sub_agent_name} analysis:
                Target: {analysis['target']}
                Trust Level: {analysis['trust_level']:.2f}
                Suspicion Level: {analysis['suspicion_level']:.2f}
                Reason: {analysis['reason']}
                """
    
    # Create analysis content for main agent
    analysis_content = Content(
        parts=[Part(text=analysis_prompt)],
        role="user"
    )
    
    # Create a new session for main agent analysis
    analysis_session_id = f"session_{agent.name}"
    
    try:
        # Get existing session or create new one
        session = session_service.get_session(
            app_name="agent_conversation",
            user_id="group_chat",
            session_id=analysis_session_id
        )
        
        if not session:
            session = session_service.create_session(
                app_name="agent_conversation",
                user_id="group_chat",
                session_id=analysis_session_id,
                state={"status": "initialized"}
            )
            
        if not session:
            raise ValueError(f"Failed to create/get session: {analysis_session_id}")
            
        # Get analysis from main agent
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
            
    except Exception as e:
        yield agent.name, f"Error in analysis: {str(e)}"
