# Grounding with Google Search

A simple yet powerful search tool that leverages Google's Gemini AI with Search Grounding to provide comprehensive, up-to-date information on any topic. Built with the latest Google Gen AI SDK (`google-genai`).

## 🌟 Features

- Uses Gemini 2.0 Flash model for fast, accurate responses
- Integrates Google Search grounding for real-time information
- Generates conversational, easy-to-understand research results
- Automatically saves results with unique identifiers
- Includes Manila time (GMT+8) timestamps
- Verifies search usage and grounding

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- Google AI Studio API key ([Get it here](https://aistudio.google.com/app/u/0/apikey))

### Installation

1. Clone the repository:
```bash
git clone https://github.com/llegomark/google-search-grounding.git
cd google-search-grounding
```

2. Install the required packages:
```bash
pip install google-genai python-dotenv pytz
```

3. Create a `.env` file in the project root and add your API key:
```bash
GOOGLE_API_KEY=your_api_key_here
```

### Usage

Run the script:
```bash
python research_tool.py
```

Simply enter any topic you want to research, and the tool will:
- Generate comprehensive research using Gemini AI
- Ground the information using Google Search
- Save results automatically to `research_results/` directory
- Display results in the console