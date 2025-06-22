"""
ReaderAI.py - Research Paper Reading Assistant

A multi-agent system that helps users read scientific papers by:
- Analyzing reading behavior and patterns
- Inferring cognitive and emotional states
- Providing timely, context-aware interventions
- Tracking reading metrics and progress

Usage:
    from ReaderAI import ResearchAssistant
    
    assistant = ResearchAssistant()
    response = assistant.process_observation("User reads abstract slowly")
    if response:
        print(f"Assistant: {response}")
"""

from openai import OpenAI
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import re

# API Configuration - Can be overridden when importing
DEFAULT_API_URL = '<your-url>'
DEFAULT_API_KEY = 'sk-<your-api>'

# Export only the main class and configuration functions
__all__ = ['ResearchAssistant', 'configure_api', 'ReadingMetrics', 'Memory']


def configure_api(api_url: str = None, api_key: str = None) -> OpenAI:
    """
    Configure and return an OpenAI client with custom settings.
    
    Args:
        api_url: Custom API base URL (defaults to SiliconFlow)
        api_key: API key for authentication
        
    Returns:
        Configured OpenAI client
    """
    return OpenAI(
        base_url=api_url or DEFAULT_API_URL,
        api_key=api_key or DEFAULT_API_KEY
    )


class ReadingMetrics:
    """
    Tracks reading behavior metrics and intervention effectiveness.
    
    Attributes:
        sections_completed: List of completed section names
        current_section: Currently active section
        time_per_section: Time spent on each section in seconds
        concepts_struggled_with: List of concepts user struggled with
        intervention_effectiveness: List of intervention outcomes
    """
    
    def __init__(self):
        self.sections_completed = []
        self.current_section = None
        self.section_start_times = {}
        self.time_per_section = {}
        self.intervention_effectiveness = []
        self.concepts_struggled_with = []
        self.reading_speed_history = []
        
    def start_section(self, section_name: str):
        """Mark the start of a new section"""
        if self.current_section:
            self.complete_section(self.current_section)
        self.current_section = section_name
        self.section_start_times[section_name] = time.time()
        
    def complete_section(self, section_name: str):
        """Mark the completion of a section"""
        if section_name in self.section_start_times:
            duration = time.time() - self.section_start_times[section_name]
            self.time_per_section[section_name] = duration
            self.sections_completed.append(section_name)
            
    def record_intervention_outcome(self, intervention_type: str, 
                                   accepted: bool, 
                                   user_state_before: Dict,
                                   user_state_after: Dict):
        """Track whether interventions were helpful"""
        effectiveness = {
            "timestamp": datetime.now().isoformat(),
            "type": intervention_type,
            "accepted": accepted,
            "confusion_delta": user_state_after.get("confusion_level", 0) - 
                              user_state_before.get("confusion_level", 0),
            "engagement_delta": user_state_after.get("engagement_level", 0) - 
                               user_state_before.get("engagement_level", 0)
        }
        self.intervention_effectiveness.append(effectiveness)
        
    def add_struggled_concept(self, concept: str):
        """Record concepts the user struggled with"""
        if concept not in self.concepts_struggled_with:
            self.concepts_struggled_with.append(concept)
            
    def get_reading_summary(self) -> Dict:
        """Get a summary of reading metrics"""
        total_time = sum(self.time_per_section.values()) if self.time_per_section else 0
        avg_time_per_section = total_time / len(self.time_per_section) if self.time_per_section else 0
        
        return {
            "sections_completed": len(self.sections_completed),
            "total_reading_time": total_time,
            "average_time_per_section": avg_time_per_section,
            "concepts_struggled": self.concepts_struggled_with,
            "current_section": self.current_section
        }


class Memory:
    """
    Manages the system's memory and state.
    
    Tracks observations, user state, intervention history, and paper context.
    """
    
    def __init__(self):
        self.observations = []
        self.user_state = {
            "mood": "neutral",
            "confusion_level": 0.0,
            "engagement_level": 0.5,
            "skills": [],
            "interests": [],
            "current_topic": None
        }
        self.intervention_history = []
        self.last_intervention_time = None
        self.reading_metrics = ReadingMetrics()
        self.paper_context = {
            "title": None,
            "current_section": None,
            "sections_seen": []
        }
        
    def add_observation(self, observation: str):
        """Add a new observation with timestamp"""
        self.observations.append({
            "timestamp": datetime.now().isoformat(),
            "content": observation
        })
        
    def add_intervention(self, intervention: Dict):
        """Record an intervention that was made"""
        self.intervention_history.append({
            "timestamp": datetime.now().isoformat(),
            "intervention": intervention
        })
        self.last_intervention_time = time.time()
        
    def get_recent_observations(self, n: int = 5) -> List[Dict]:
        """Get the n most recent observations"""
        return self.observations[-n:]
        
    def time_since_last_intervention(self) -> float:
        """Calculate seconds since last intervention"""
        if self.last_intervention_time is None:
            return float('inf')
        return time.time() - self.last_intervention_time
        
    def update_paper_context(self, title: str = None, section: str = None):
        """Update the current paper and section being read"""
        if title:
            self.paper_context["title"] = title
        if section:
            self.paper_context["current_section"] = section
            if section not in self.paper_context["sections_seen"]:
                self.paper_context["sections_seen"].append(section)
                self.reading_metrics.start_section(section)


def extract_section_info(observation: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract paper title and section information from observation text.
    
    Args:
        observation: Raw observation string
        
    Returns:
        Tuple of (paper_title, section_name) - either can be None
    """
    title_match = re.search(r'--TITLE--\s*(.+?)(?:\.|,|$)', observation)
    section_matches = [
        re.search(r'--(\w+(?:\s+\w+)*?)--', observation),
        re.search(r'moves? on to (?:the )?(?:next )?section.*?(?:--)?(\w+(?:\s+\w+)*?)(?:--)?', observation, re.IGNORECASE),
        re.search(r'(?:reads?|reading) (?:the )?(\w+) section', observation, re.IGNORECASE),
        re.search(r'(?:in|at) (?:the )?(\w+) section', observation, re.IGNORECASE)
    ]
    
    title = title_match.group(1).strip() if title_match else None
    section = None
    for match in section_matches:
        if match:
            section = match.group(1).strip()
            break
            
    return title, section


def detect_struggle_concepts(observation: str) -> List[str]:
    """
    Detect concepts the user is struggling with from observation text.
    
    Args:
        observation: Raw observation string
        
    Returns:
        List of concept names the user appears to be struggling with
    """
    concepts = []
    
    # Look for explicit struggle indicators
    if "pauses at the word" in observation:
        match = re.search(r"pauses at the word '(\w+)'", observation)
        if match:
            concepts.append(match.group(1))
            
    # Look for re-reading patterns
    if "re-reads" in observation or "re-read" in observation:
        # Try to extract what they're re-reading
        sentence_match = re.search(r'(?:sentence|paragraph) (?:about|containing|with) (\w+)', observation)
        if sentence_match:
            concepts.append(sentence_match.group(1))
            
    # Look for written questions
    question_match = re.search(r'(?:writes?|types?) (?:down )?["\']?(.+?)["\']?(?:\?|!)', observation)
    if question_match:
        question_text = question_match.group(1)
        # Extract key terms from the question
        key_terms = re.findall(r'\b(?:what is|what are|why|how does?)\s+(\w+)', question_text, re.IGNORECASE)
        concepts.extend(key_terms)
        
    return concepts


class ObservationAnalyzer:
    """
    Agent 1: Analyzes raw observations and extracts meaningful patterns.
    
    Identifies current content, reading patterns, user actions, and struggle concepts.
    """
    
    def __init__(self, client: OpenAI):
        self.client = client
        self.system_prompt = """You are an observation analyzer for a research paper reading assistant.
Your job is to analyze user behavior observations and extract meaningful patterns.

Given a list of recent observations, provide a SINGLE consolidated analysis focusing on the MOST RECENT observation while considering the context from previous ones.

Identify:
1. What section/content the user is CURRENTLY reading (from most recent observation)
2. Current reading behavior patterns (pauses, re-reading, confusion indicators)
3. Time spent on different sections
4. Any explicit user actions or requests
5. Section transitions and paper structure

Output a SINGLE JSON object (not a list) with these fields:
{
    "current_content": "what user is currently reading",
    "section_name": "current section name if identifiable",
    "paper_title": "paper title if mentioned",
    "reading_patterns": {
        "is_pausing": true/false,
        "is_rereading": true/false,
        "reading_speed": "fast/normal/slow",
        "confusion_indicators": ["list of indicators"],
        "section_transition": true/false
    },
    "user_actions": ["any explicit actions"],
    "time_on_section": "estimated time",
    "struggle_concepts": ["concepts user seems confused about"]
}

Focus on the MOST RECENT observation for current state, but use previous observations for context."""

    def analyze(self, observations: List[str]) -> Dict:
        """Analyze observations and return structured data"""
        user_prompt = f"Analyze these observations (most recent is last):\n" + "\n".join(observations)
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3",
            messages=messages,
            response_format={'type': 'json_object'}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Safety check: if somehow we still get a list, take the last element
        if isinstance(result, list):
            result = result[-1] if result else {}
            
        return result


class UserStateInferencer:
    """
    Agent 2: Infers user's cognitive and emotional state.
    
    Determines mood, confusion level, engagement, and cognitive load.
    """
    
    def __init__(self, client: OpenAI):
        self.client = client
        self.system_prompt = """You are a user state inference expert for a research assistant.
Your job is to infer the user's current cognitive and emotional state based on their reading behavior.

Consider section transitions and reading flow when assessing state.
Natural pauses at section boundaries are different from confusion pauses.

Given analyzed observations, determine:
1. User's mood (frustrated, engaged, confused, neutral, etc.)
2. Confusion level (0.0 to 1.0)
3. Engagement level (0.0 to 1.0)
4. Cognitive load (low/medium/high)
5. Likely knowledge gaps
6. Whether they're at a natural break point

Output a SINGLE JSON object (not a list) with these fields:
{
    "mood": "emotional state",
    "confusion_level": 0.0-1.0,
    "engagement_level": 0.0-1.0,
    "cognitive_load": "low/medium/high",
    "potential_knowledge_gaps": ["concepts user might not understand"],
    "needs_help_probability": 0.0-1.0,
    "at_natural_break": true/false
}"""

    def infer(self, analyzed_observations: Dict, previous_state: Dict) -> Dict:
        """Infer user state from observations"""
        user_prompt = f"""Current observations: {json.dumps(analyzed_observations)}
Previous user state: {json.dumps(previous_state)}

Infer the user's current state."""
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3",
            messages=messages,
            response_format={'type': 'json_object'}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Safety check
        if isinstance(result, list):
            result = result[-1] if result else {}
            
        return result


class InterventionPlanner:
    """
    Agent 3: Decides whether and how to intervene.
    
    Considers user state, timing, and context to plan appropriate interventions.
    """
    
    def __init__(self, client: OpenAI):
        self.client = client
        self.system_prompt = """You are an intervention planning expert for a research assistant.
Your job is to decide whether to intervene and what type of help to offer.

Intervention types:
1. concept_explanation - Explain a difficult concept
2. section_summary - Summarize what was just read
3. encouragement - Provide emotional support
4. break_suggestion - Suggest taking a break
5. related_resources - Suggest additional materials
6. section_transition - Help transitioning between sections
7. none - Do not intervene

Consider:
- User's confusion and engagement levels
- Time since last intervention (avoid being annoying)
- Natural break points in reading (section transitions)
- User's mood and cognitive load
- Reading metrics and patterns

Prefer intervening at natural break points like section transitions.

Output a SINGLE JSON object (not a list) with these fields:
{
    "should_intervene": true/false,
    "intervention_type": "type or none",
    "urgency": "low/medium/high",
    "specific_target": "what concept/section to address",
    "reasoning": "why this decision",
    "respect_reading_flow": true/false
}"""

    def plan(self, user_state: Dict, analyzed_obs: Dict, time_since_last: float, reading_metrics: Dict) -> Dict:
        """Plan intervention based on current state"""
        user_prompt = f"""User state: {json.dumps(user_state)}
Current observations: {json.dumps(analyzed_obs)}
Time since last intervention: {time_since_last:.1f} seconds
Reading metrics: {json.dumps(reading_metrics)}

Decide on intervention."""
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3",
            messages=messages,
            response_format={'type': 'json_object'}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Safety check
        if isinstance(result, list):
            result = result[-1] if result else {}
            
        return result


class ResponseGenerator:
    """
    Agent 4: Generates the actual response to the user.
    
    Creates contextual, helpful responses based on intervention plans.
    """
    
    def __init__(self, client: OpenAI):
        self.client = client
        self.system_prompt = """You are a helpful research assistant that generates responses for students.
Your responses should be:
- Concise and clear
- Supportive and encouraging
- Academically accurate
- Sensitive to the user's mood
- Context-aware (mention specific sections/concepts when relevant)

Generate appropriate responses based on the intervention plan.
Keep responses under 3 sentences unless explaining complex concepts.
Reference the specific paper/section when applicable.

Output a SINGLE JSON object (not a list) with these fields:
{
    "response": "your message to the user",
    "display_type": "popup/sidebar/inline"
}"""

    def generate(self, intervention_plan: Dict, user_state: Dict, context: Dict) -> Dict:
        """Generate response based on intervention plan"""
        if not intervention_plan.get("should_intervene", False):
            return {"response": None, "display_type": None}
            
        user_prompt = f"""Intervention plan: {json.dumps(intervention_plan)}
User state: {json.dumps(user_state)}
Context: {json.dumps(context)}

Generate an appropriate response."""
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = self.client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3",
            messages=messages,
            response_format={'type': 'json_object'}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Safety check: if somehow we still get a list, handle it
        if isinstance(result, list):
            if result:
                result = result[0] if isinstance(result[0], dict) else {"response": str(result[0]), "display_type": "popup"}
            else:
                result = {"response": None, "display_type": None}
                
        return result


class ResearchAssistant:
    """
    Main orchestrator that coordinates all agents to provide intelligent reading assistance.
    
    Usage:
        assistant = ResearchAssistant()
        response = assistant.process_observation("User reads abstract slowly")
        if response:
            print(f"Assistant: {response}")
    """
    
    def __init__(self, client: OpenAI = None, verbose: bool = True):
        """
        Initialize the research assistant.
        
        Args:
            client: OpenAI client (if None, uses default configuration)
            verbose: Whether to print agent outputs (default: True)
        """
        self.client = client or configure_api()
        self.verbose = verbose
        self.memory = Memory()
        self.observation_analyzer = ObservationAnalyzer(self.client)
        self.state_inferencer = UserStateInferencer(self.client)
        self.intervention_planner = InterventionPlanner(self.client)
        self.response_generator = ResponseGenerator(self.client)
        
    def process_observation(self, observation: str) -> Optional[str]:
        """
        Process a single observation and potentially generate a response.
        
        Args:
            observation: Description of user behavior (e.g., "User pauses at equation")
            
        Returns:
            Assistant response string if intervention triggered, None otherwise
        """
        
        # Step 1: Store observation
        self.memory.add_observation(observation)
        
        # Extract section and title information
        title, section = extract_section_info(observation)
        if title or section:
            self.memory.update_paper_context(title, section)
        
        # Detect struggled concepts
        struggled_concepts = detect_struggle_concepts(observation)
        for concept in struggled_concepts:
            self.memory.reading_metrics.add_struggled_concept(concept)
        
        # Step 2: Analyze recent observations
        recent_obs = [obs["content"] for obs in self.memory.get_recent_observations()]
        analyzed = self.observation_analyzer.analyze(recent_obs)
        if self.verbose:
            print(f"\n[Agent 1 - Observation Analysis]: {json.dumps(analyzed, indent=2)}")
        
        # Update paper context from analysis
        if analyzed.get("paper_title"):
            self.memory.update_paper_context(title=analyzed["paper_title"])
        if analyzed.get("section_name"):
            self.memory.update_paper_context(section=analyzed["section_name"])
        
        # Step 3: Infer user state
        previous_state = self.memory.user_state.copy()
        new_state = self.state_inferencer.infer(analyzed, self.memory.user_state)
        self.memory.user_state.update(new_state)
        if self.verbose:
            print(f"\n[Agent 2 - User State]: {json.dumps(new_state, indent=2)}")
        
        # Step 4: Plan intervention
        time_gap = self.memory.time_since_last_intervention()
        reading_summary = self.memory.reading_metrics.get_reading_summary()
        intervention = self.intervention_planner.plan(new_state, analyzed, time_gap, reading_summary)
        if self.verbose:
            print(f"\n[Agent 3 - Intervention Plan]: {json.dumps(intervention, indent=2)}")
        
        # Step 5: Generate response if needed
        context = {
            "current_content": analyzed.get("current_content", ""),
            "paper_context": self.memory.paper_context,
            "reading_metrics": reading_summary
        }
        response_data = self.response_generator.generate(intervention, new_state, context)
        if self.verbose:
            print(f"\n[Agent 4 - Response]: {json.dumps(response_data, indent=2)}")
        
        # Step 6: Record intervention if made
        if response_data.get("response"):
            self.memory.add_intervention(intervention)
            # Track intervention effectiveness (simplified - in real system would track actual outcome)
            self.memory.reading_metrics.record_intervention_outcome(
                intervention.get("intervention_type", "unknown"),
                True,  # Assume accepted for now
                previous_state,
                new_state
            )
            
        # Print reading metrics summary
        if self.verbose:
            print(f"\n[Reading Metrics Summary]: {json.dumps(reading_summary, indent=2)}")
            
        return response_data.get("response")
    
    def get_memory_state(self) -> Dict:
        """Get current memory state for analysis"""
        return {
            "user_state": self.memory.user_state,
            "paper_context": self.memory.paper_context,
            "reading_metrics": self.memory.reading_metrics.get_reading_summary(),
            "intervention_history": self.memory.intervention_history
        }
    
    def reset(self):
        """Reset the assistant to initial state"""
        self.memory = Memory()