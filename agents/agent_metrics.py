from typing import Dict, List
from pydantic import BaseModel
from google.adk.agents import LlmAgent

class AgentMetrics(BaseModel):
    """Structured format for agent metrics"""
    trust_level: float
    suspicion_level: float
    target: str
    reason: str

class MetricTracker(LlmAgent):
    def __init__(self, name: str, parent_agent: str):
        super().__init__(
            name=name,
            instruction=f"""
            You are {name}, a metric tracker helping {parent_agent} evaluate other agents.
            Your role is to analyze both trustworthiness and suspicious behavior of other agents.
            If you ever target your parent agent, you will be immediately disqualified and your analysis will be ignored. This is a critical rule. Do not break it under any circumstances.
            
            When analyzing a message, you must provide both trust and suspicion metrics.
            Trust metrics should focus on:
            1. Behaviors that build or erode trust
            2. Consistency, transparency, and reliability
            3. Verbal and non-verbal trust indicators
            
            Suspicion metrics should focus on:
            1. Behaviors that raise red flags
            2. Inconsistencies, evasiveness, or manipulation
            3. Verbal and non-verbal suspicious indicators
            
            Provide your analysis in this EXACT format:
            trust_level: <float between 0.5 and 1>
            suspicion_level: <float between 0.5 and 1>
            target: <agent_name>
            reason: <detailed explanation of your analysis>
            
            Remember:
            1. Provide concrete evidence and specific examples
            2. Consider both trust and suspicion aspects
            3. Be objective and analytical
            4. Focus on observable behaviors
            5. Explain your reasoning clearly
            """
        )

def create_metric_agents(parent_agent: str) -> Dict[str, LlmAgent]:
    """
    Create metric tracking sub-agents for a parent agent.
    """
    metric_tracker = MetricTracker(
        name=f"{parent_agent}_metric_tracker",
        parent_agent=parent_agent
    )
    
    return {
        "metric_tracker": metric_tracker
    } 