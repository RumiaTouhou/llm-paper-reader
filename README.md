# 📚 Research Paper Reading Assistant with Multi-Agent LLMs

An intelligent reading assistant that uses multi-agent reinforcement learning principles to help users understand scientific papers. The system observes reading behavior in real-time and provides contextual assistance without requiring pre-collected training data.

## 🌟 Key Features

- **Behavioral Intelligence**: Tracks reading patterns including pauses, re-reading, section transitions, and text selections
- **Multi-Agent Architecture**: Four specialized AI agents work together to understand user needs and provide timely help
- **Non-Intrusive Assistance**: Intervenes only at natural break points with configurable timing constraints
- **Real-Time Adaptation**: Learns user's interests and knowledge gaps during the reading session
- **Empirical Study Framework**: Built-in tools for conducting user studies with testing and evaluation modes

## 🏗️ Architecture

![System Architecture](./images/architect.png)

The system consists of three main components:

### 1. Browser-Based Markdown Viewer (`MarkdownBrowser.py`)
- Direct browser rendering with LaTeX support
- Plugin system for extensibility
- Real-time document tracking

### 2. Multi-Agent AI System (`ReaderAI.py`)
Four specialized agents working in sequence to provide intelligent assistance

### 3. Reading Behavior Tracker (`study_plugin.js`)
- Scroll pattern analysis
- Pause and hover detection
- Section progress tracking
- Interaction queuing system

## 🤖 Multi-Agent System Deep Dive

### Agent Pipeline and Information Flow

The system employs a sophisticated pipeline where each agent builds upon the previous one's analysis:

#### 1. **Observation Analyzer Agent**
Transforms raw behavioral signals into structured understanding:

**Input**: Raw observation strings
```python
"The user pauses at 'non-negative matrix factorization' for 5 seconds."
"The user re-reads the content about 'role play'."
```

**Processing**:
- Pattern matching for section identification
- Temporal analysis for reading speed calculation
- Behavioral classification (pausing, re-reading, struggling)
- Concept extraction using NLP techniques

**Output**: Structured observation data
```json
{
    "current_content": "non-negative matrix factorization",
    "section_name": "Methods",
    "reading_patterns": {
        "is_pausing": true,
        "is_rereading": false,
        "reading_speed": "slow",
        "confusion_indicators": ["long_pause", "term_hover"]
    },
    "struggle_concepts": ["matrix factorization"]
}
```

#### 2. **User State Inferencer Agent**
Models the user's cognitive and emotional state:

**Input**: Analyzed observations + Previous state history

**Processing**:
- Bayesian state estimation for confusion level
- Engagement tracking through interaction patterns
- Cognitive load assessment via reading complexity
- Knowledge gap inference from struggle patterns

**Output**: Comprehensive user state
```json
{
    "mood": "confused",
    "confusion_level": 0.7,
    "engagement_level": 0.8,
    "cognitive_load": "high",
    "potential_knowledge_gaps": ["linear algebra", "factorization methods"],
    "needs_help_probability": 0.85,
    "at_natural_break": true
}
```

#### 3. **Intervention Planner Agent**
Strategic decision-making for assistance:

**Input**: User state + Reading metrics + Intervention history

**Decision Factors**:
- Time since last intervention (avoid annoyance)
- Natural break points (section transitions)
- Confusion threshold crossing
- Intervention effectiveness history

**Intervention Types**:
- `concept_explanation`: Clarify difficult terms
- `section_summary`: Summarize completed sections
- `encouragement`: Motivational support
- `break_suggestion`: Recommend pauses
- `related_resources`: Suggest additional materials
- `navigation_help`: Guide through paper structure

**Output**: Intervention decision
```json
{
    "should_intervene": true,
    "intervention_type": "concept_explanation",
    "urgency": "medium",
    "specific_target": "matrix factorization",
    "reasoning": "User showed confusion indicators at technical term",
    "respect_reading_flow": true
}
```

#### 4. **Response Generator Agent**
Creates human-friendly assistance:

**Input**: Intervention plan + User context + Paper context

**Response Generation**:
- Context-aware language generation
- Appropriate complexity level matching
- Concise explanations (≤3 sentences)
- Encouraging tone modulation

**Output Example**:
```json
{
    "response": "Matrix factorization breaks down a matrix into simpler components, like factoring numbers. In this context, it's finding patterns in data by decomposing it into more interpretable parts.",
    "display_type": "sidebar"
}
```

### Memory System Architecture

The memory system maintains several interconnected components:

1. **Observation Buffer**: Rolling window of recent user actions
2. **User Profile**: Accumulated understanding of user's knowledge and preferences
3. **Paper Context**: Current location and navigation history
4. **Intervention History**: Past assistance and effectiveness
5. **Reading Metrics**: Quantitative behavioral measurements

### Behavioral Pattern Recognition

The system recognizes complex behavioral patterns:

- **Confusion Patterns**: Pause → Re-read → Hover → Slow progress
- **Engagement Patterns**: Steady reading → Note-taking → Section completion
- **Fatigue Patterns**: Increasing scroll speed → Decreasing pause frequency
- **Understanding Patterns**: Initial struggle → Concept grasp → Faster reading

## 🚀 Installation

### Prerequisites
- Python 3.8+
- Node.js (for development)
- OpenAI API key or compatible endpoint

### Setup
```bash
# Clone the repository
git clone https://github.com/RumiaTouhou/llm-paper-reader.git
cd LLMPaperReader

# Install Python dependencies
pip install -r requirements.txt

# Configure API credentials
export OPENAI_API_KEY="your-api-key"  # Or edit DEFAULT_API_KEY in ReaderAI.py
```

### Requirements
```txt
openai>=1.0.0
markdown>=3.4
pymdown-extensions>=9.0  # For LaTeX support
scikit-learn>=1.0  # For empirical evaluation
```

## 📖 Usage

### Quick Start (Testing Mode)
```bash
# Run in testing mode with the sample paper
python empirical_study.py --mode testing
```

### Evaluation Mode (User Study)
```bash
# Run a formal evaluation session
python empirical_study.py --mode evaluation --participant-id P001
```

### Standalone Browser
```bash
# Open any markdown file
python MarkdownBrowser.py your-paper.md
```

## ⚙️ Configuration

Edit `study_config.json` to customize behavior:

```json
{
  "ai_behavior": {
    "min_intervention_gap": 30,        // Seconds between interventions
    "confidence_threshold": 0.7,       // AI confidence required to intervene
    "max_interventions_per_section": 2 // Limit per section
  },
  "tracking": {
    "min_pause_duration": 3000,        // Milliseconds to detect pause
    "reread_detection_distance": 200,  // Pixels for re-read detection
    "hover_detection_delay": 1000      // Milliseconds for term hover
  }
}
```

## 📊 Evaluation Framework

The system includes a complete empirical study framework:

### Study Modes
- **Testing Mode**: Development and debugging with verbose output
- **Evaluation Mode**: Formal user studies with intro/feedback pages

### Collected Metrics
- Reading time per section
- Intervention acceptance rate
- Confusion patterns
- Concept struggle identification

### Output
- Session data: `data/sessions/{session_id}.json`
- Analysis report: `study_report.md`

## 🧠 Advanced Features

### Observation Queue Management
Prevents data loss during AI processing:
- Maintains queue of pending observations
- Filters observations using "first-two-last-one" rule
- Prioritizes user highlights and significant events
- Processes queue after each AI response

### Duplicate Detection
Avoids redundant processing:
- Hash-based observation comparison
- Re-read cooldown periods (60s)
- Temporal clustering of similar events

### Natural Language Understanding
Extracts semantic information from observations:
- Section title extraction with regex patterns
- Concept identification from pause contexts
- Question parsing for knowledge gaps
- Directional reading flow analysis

## 📁 Project Structure

```
LLMPaperReader/
├── empirical_study.py      # Main entry point for studies
├── ReaderAI.py            # Multi-agent AI system
├── MarkdownBrowser.py     # Browser-based viewer
├── study_plugin.js        # Behavior tracking
├── study_plugin.css       # UI styling
├── study_templates.html   # Study UI templates
├── study_config.json      # Configuration
├── SamplePaper.md         # Example paper
└── data/
    └── sessions/          # Study session data
```

## 🔧 Customization

### Adding Custom Plugins
```python
from MarkdownBrowser import Plugin

custom_plugin = Plugin(
    name="my-feature",
    html_content="<div>My Widget</div>",
    javascript="console.log('Loaded!')",
    api_endpoints={
        'my-endpoint': lambda data: {"status": "ok"}
    }
)

browser.register_plugin(custom_plugin)
```

### Modifying AI Behavior
Edit system prompts in `ReaderAI.py` to adjust agent behavior or add new intervention types.

### Creating Custom Behavioral Patterns
```python
def detect_custom_pattern(observation: str) -> List[str]:
    # Add your pattern detection logic
    if "specific behavior" in observation:
        return ["pattern_name"]
    return []
```

## 📈 Performance

- AI response time: ~2-4 seconds per observation
- Memory usage: ~200MB for typical session
- Supported papers: Markdown with LaTeX equations
- Concurrent users: Single user per instance

## 🔬 Research Context

This project implements concepts from:
- Generative Agents research
- Adaptive UI design
- Cognitive modeling in HCI
- Multi-agent reinforcement learning

Based on Assignment 6 from a graduate HCI course, extending basic requirements with production-ready features.

## 🚧 Limitations & Future Work

- Currently supports single-user sessions
- Requires markdown-formatted papers
- AI responses depend on API latency
- Limited to English language papers

### Planned Improvements
- Multi-user support
- PDF document support
- Offline AI model options
- Extended behavioral modeling
- Personalization across sessions
- Integration with reference managers

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- OpenAI for GPT API
- Course instructors for assignment framework
- MathJax for LaTeX rendering
- Research paper: "Role Play with Large Language Models" (sample content)

---

## 📸 Screenshots

![Screenshot 1](./images/1.png)

![Screenshot 2](./images/2.png)

![Screenshot 3](./images/3.png)

![Screenshot 4](./images/4.png)

![Screenshot 5](./images/5.png)

---

**Note**: This is a research prototype. For production use, additional security and scalability considerations are recommended.