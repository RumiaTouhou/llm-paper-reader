#!/usr/bin/env python3
"""
Empirical Study System for Research Paper Reading Assistant

Usage:
    python empirical_study.py --mode testing
    python empirical_study.py --mode evaluation --participant-id P001
"""

import argparse
import json
import os
import time
import uuid
from datetime import datetime
from typing import Dict, Optional
import webbrowser
from http.server import BaseHTTPRequestHandler
import urllib.parse

# Import existing modules
from MarkdownBrowser import DirectMarkdownBrowser, Plugin
from ReaderAI import ResearchAssistant


class StudySession:
    """Manages a single study session"""
    
    def __init__(self, mode: str, participant_id: str = None):
        self.mode = mode
        self.participant_id = participant_id or f"test_{uuid.uuid4().hex[:8]}"
        self.session_id = f"{self.participant_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.start_time = datetime.now()
        self.interactions = []
        self.metrics = {
            "total_reading_time": 0,
            "sections_completed": 0,
            "ai_interventions": 0,
            "interventions_accepted": 0,
            "interventions_rejected": 0,
            "pauses_detected": 0,
            "rereading_detected": 0
        }
        
        # Initialize AI assistant
        self.assistant = ResearchAssistant(verbose=False)
        
        # Create data directory
        os.makedirs("data/sessions", exist_ok=True)
        
    def log_interaction(self, interaction_type: str, data: Dict):
        """Log an interaction with timestamp"""
        self.interactions.append({
            "timestamp": datetime.now().isoformat(),
            "type": interaction_type,
            "data": data
        })
        
    def process_observation(self, observation: str) -> Optional[Dict]:
        """Process observation through AI and return response"""
        response = self.assistant.process_observation(observation)
        
        if response:
            self.metrics["ai_interventions"] += 1
            self.log_interaction("ai_response", {
                "observation": observation,
                "response": response
            })
            return {"response": response, "type": "suggestion"}
        
        return None
        
    def record_feedback(self, helpful: bool):
        """Record user feedback on intervention"""
        if helpful:
            self.metrics["interventions_accepted"] += 1
        else:
            self.metrics["interventions_rejected"] += 1
            
        self.log_interaction("user_feedback", {"helpful": helpful})
        
    def save_session(self):
        """Save session data to file"""
        self.metrics["total_reading_time"] = (datetime.now() - self.start_time).total_seconds()
        
        session_data = {
            "session_id": self.session_id,
            "participant_id": self.participant_id,
            "mode": self.mode,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "metrics": self.metrics,
            "interactions": self.interactions,
            "ai_memory": self.assistant.get_memory_state()
        }
        
        filename = f"data/sessions/{self.session_id}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2)
            
        print(f"Session saved to: {filename}")
        return filename
        
    def generate_report(self):
        """Generate analysis report for evaluation mode"""
        if self.mode != "evaluation":
            return
            
        report = f"""# Empirical Study Report

## Session Information
- **Participant ID**: {self.participant_id}
- **Session ID**: {self.session_id}
- **Date**: {self.start_time.strftime('%Y-%m-%d %H:%M')}
- **Duration**: {self.metrics['total_reading_time']:.1f} seconds

## Reading Metrics
- **Sections Completed**: {self.metrics['sections_completed']}
- **Pauses Detected**: {self.metrics['pauses_detected']}
- **Re-reading Instances**: {self.metrics['rereading_detected']}

## AI Assistant Performance
- **Total Interventions**: {self.metrics['ai_interventions']}
- **Accepted**: {self.metrics['interventions_accepted']}
- **Rejected**: {self.metrics['interventions_rejected']}
- **Acceptance Rate**: {self.metrics['interventions_accepted'] / max(1, self.metrics['ai_interventions']) * 100:.1f}%

## Interaction Timeline
"""
        
        # Add significant interactions
        for interaction in self.interactions[-20:]:  # Last 20 interactions
            if interaction['type'] in ['ai_response', 'user_feedback']:
                report += f"\n- **{interaction['timestamp']}**: {interaction['type']}"
                if 'observation' in interaction.get('data', {}):
                    report += f"\n  - Observation: {interaction['data']['observation'][:100]}..."
                    
        with open("study_report.md", 'w', encoding='utf-8') as f:
            f.write(report)
            
        print("Report generated: study_report.md")


class StudyBridge:
    """Bridge between browser and AI assistant"""
    
    def __init__(self, session: StudySession, config: Dict):
        self.session = session
        self.config = config
        self.last_observation_time = time.time()
        
    def handle_observation(self, data: Dict) -> Dict:
        """Handle observation from browser"""
        observation_text = data.get('observation', '')
        observation_type = data.get('type', 'general')
        
        # Add debug logging
        print(f"[Bridge] Received observation: {observation_type} - {observation_text[:50]}...")
        
        # Log the raw observation
        self.session.log_interaction("user_behavior", data)
        
        # Update metrics based on observation type
        if observation_type == 'pause':
            self.session.metrics['pauses_detected'] += 1
        elif observation_type == 'reread':
            self.session.metrics['rereading_detected'] += 1
        elif observation_type == 'section_complete':
            self.session.metrics['sections_completed'] += 1
            
        # Check if enough time has passed since last observation
        current_time = time.time()
        time_gap = current_time - self.last_observation_time
        
        # Process through AI if appropriate
        if self.config.get('ai_enabled', True) and time_gap > 2.0:  # 2 second minimum gap
            self.last_observation_time = current_time
            
            print(f"[Bridge] Processing through AI (time gap: {time_gap:.1f}s)")
            start_time = time.time()
            
            ai_response = self.session.process_observation(observation_text)
            
            processing_time = time.time() - start_time
            print(f"[Bridge] AI processing took {processing_time:.1f}s")
            
            if ai_response:
                return ai_response
        else:
            print(f"[Bridge] Skipping AI (time gap: {time_gap:.1f}s, AI enabled: {self.config.get('ai_enabled', True)})")
                
        return {"response": None}
        
    def handle_feedback(self, data: Dict) -> Dict:
        """Handle user feedback on AI response"""
        helpful = data.get('helpful', False)
        self.session.record_feedback(helpful)
        return {"status": "recorded"}
        
    def handle_session_control(self, data: Dict) -> Dict:
        """Handle session control commands"""
        command = data.get('command', '')
        
        if command == 'end_session':
            filename = self.session.save_session()
            if self.session.mode == 'evaluation':
                self.session.generate_report()
            
            # Signal to close the browser
            return {
                "status": "completed", 
                "filename": filename,
                "should_close": True,
                "message": "Study session completed. You can close the browser window."
            }
            
        return {"status": "unknown_command"}


def create_study_plugin(bridge: StudyBridge, config: Dict) -> Plugin:
    """Create the browser plugin for the study"""
    
    # Load assets
    with open('study_plugin.js', 'r', encoding='utf-8') as f:
        javascript = f.read()
    with open('study_plugin.css', 'r', encoding='utf-8') as f:
        css = f.read()
    with open('study_templates.html', 'r', encoding='utf-8') as f:
        templates_html = f.read()
        
    # Extract assistant widget template
    import re
    widget_match = re.search(r'<!-- TEMPLATE: assistant_widget -->(.+?)<!-- END TEMPLATE -->', 
                           templates_html, re.DOTALL)
    widget_html = widget_match.group(1).strip() if widget_match else ""
    
    # Configure JavaScript with settings
    js_config = f"""
    // Injected configuration
    window.STUDY_CONFIG = {json.dumps(config)};
    window.STUDY_MODE = '{bridge.session.mode}';
    window.PARTICIPANT_ID = '{bridge.session.participant_id}';
    
    {javascript}
    """
    
    # API endpoints
    def observation_endpoint(data):
        return bridge.handle_observation(data)
        
    def feedback_endpoint(data):
        return bridge.handle_feedback(data)
        
    def control_endpoint(data):
        return bridge.handle_session_control(data)
    
    return Plugin(
        name="reading-study",
        html_content=widget_html,
        javascript=js_config,
        css=css,
        api_endpoints={
            'observe': observation_endpoint,
            'feedback': feedback_endpoint,
            'control': control_endpoint
        }
    )


def show_intro_page(mode: str):
    """Show introduction page for evaluation mode"""
    if mode != 'evaluation':
        return
        
    with open('study_templates.html', 'r') as f:
        templates = f.read()
        
    # Extract intro template
    import re
    intro_match = re.search(r'<!-- TEMPLATE: intro -->(.+?)<!-- END TEMPLATE -->', 
                          templates, re.DOTALL)
    
    if intro_match:
        intro_html = intro_match.group(1).strip()
        
        # Create temporary HTML file
        with open('temp_intro.html', 'w') as f:
            f.write(intro_html)
            
        webbrowser.open('file://' + os.path.abspath('temp_intro.html'))
        
        print("\n📋 Please read the introduction page in your browser.")
        input("Press Enter when ready to begin the study...")
        
        # Clean up
        os.remove('temp_intro.html')


def show_feedback_page(session_id: str):
    """Show feedback page after evaluation"""
    with open('study_templates.html', 'r') as f:
        templates = f.read()
        
    # Extract feedback template
    import re
    feedback_match = re.search(r'<!-- TEMPLATE: feedback -->(.+?)<!-- END TEMPLATE -->', 
                             templates, re.DOTALL)
    
    if feedback_match:
        feedback_html = feedback_match.group(1).strip()
        feedback_html = feedback_html.replace('{{SESSION_ID}}', session_id)
        
        # Create temporary HTML file
        with open('temp_feedback.html', 'w') as f:
            f.write(feedback_html)
            
        webbrowser.open('file://' + os.path.abspath('temp_feedback.html'))
        
        print("\n📋 Please complete the feedback form in your browser.")
        print("You can close this terminal when done.")


def main():
    parser = argparse.ArgumentParser(description='Research Paper Reading Assistant Empirical Study')
    parser.add_argument('--mode', choices=['testing', 'evaluation'], required=True,
                       help='Study mode: testing or evaluation')
    parser.add_argument('--participant-id', type=str, 
                       help='Participant ID (required for evaluation mode)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == 'evaluation' and not args.participant_id:
        parser.error("--participant-id is required for evaluation mode")
        
    # Load configuration
    with open('study_config.json', 'r') as f:
        config = json.load(f)
        
    mode_config = config['modes'][args.mode]
    
    # Show intro for evaluation mode
    if mode_config.get('show_intro', False):
        show_intro_page(args.mode)
        
    # Create session
    session = StudySession(args.mode, args.participant_id)
    bridge = StudyBridge(session, mode_config)
    
    # Create and configure browser
    browser = DirectMarkdownBrowser()
    browser.load_markdown_file('SamplePaper.md')
    
    # Register study plugin
    study_plugin = create_study_plugin(bridge, config)
    browser.register_plugin(study_plugin)
    
    print(f"\n🚀 Starting {args.mode} mode session")
    print(f"📊 Session ID: {session.session_id}")
    print(f"👤 Participant: {session.participant_id}")
    
    try:
        # Run the browser
        browser.run()
    except KeyboardInterrupt:
        print("\n\n⏹️  Session interrupted by user")
    finally:
        # Save session data
        filename = session.save_session()
        
        # Generate report and show feedback for evaluation mode
        if args.mode == 'evaluation':
            session.generate_report()
            if mode_config.get('show_feedback', False):
                show_feedback_page(session.session_id)
                
        print(f"\n✅ Study session completed")
        print(f"📁 Data saved to: {filename}")


if __name__ == "__main__":
    main()