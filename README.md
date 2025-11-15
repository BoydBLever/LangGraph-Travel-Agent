# LangGraph Multi-Agent Travel Agent

## Demo

<video width="100%" controls>
  <source src="https://github.com/BoydBLever/LangGraph-Travel-Agent/releases/download/v0.1.0/demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## Project Overview
This project implements a **multi-agent travel planning system** using **LangGraph**, designed to assist users with various travel-related queries, including flight bookings, hotel searches, and itinerary planning. It leverages specialized agents (Itinerary, Flight, Hotel) and an intelligent routing mechanism to direct user requests to the most appropriate agent.

## Features

- **Multi-Agent Architecture**  
  Orchestrates specialized agents for different travel domains.

- **Intelligent Routing**  
  A dedicated router analyzes user queries and directs them to the correct agent (Flight, Hotel, or Itinerary).

- **Itinerary Agent**  
  Provides detailed travel plans, activity suggestions, and general travel advice using Tavily Search for real-time information.

- **Flight Agent**  
  Searches for flight options based on user criteria (departure, arrival, dates, passengers) using the SerpApi Google Flights engine.

- **Hotel Agent**  
  Finds hotel accommodations based on location, dates, and preferences using the SerpApi Google Hotels engine.

- **Stateful Conversations**  
  Utilizes LangGraph’s state management for persistent context across turns in multi-turn conversations.

- **Modular Design**  
  Code is organized into separate Python files (`state.py`, `llm_config.py`, `tools.py`, `agents.py`, `router.py`, `graph_builder.py`, `main.py`) for better maintainability and readability.

## Setup and Installation

### Prerequisites
- Python 3.9+
- pip (Python package installer)
- Git

### 1. Clone the Repository
First, clone this repository to your local machine:

git clone https://github.com/BoydBLever/LangGraph-Travel-Agent.git
cd LangGraph-Travel-Agent

### 2. Install Dependencies
It's recommended to use a virtual environment, but for direct installation, run:

pip install -r requirements.txt

**Note:** If you encounter a `ResolutionTooDeep` error during installation:

pip install --upgrade pip
pip install -r requirements.txt

### 3. Obtain API Keys
This project requires API keys for:
- **Google Gemini (Google AI Studio)** – core LLM  
- **Tavily Search** – itinerary web search  
- **SerpApi** – real-time flight & hotel search  

Get keys from https://serpapi.com , https://aistudio.google.com/ , and https://tavily.com/

### 4. Set Environment Variables
`main.py` will prompt for missing keys interactively. For a persistent setup, create a `.env` file:

GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY_HERE"  
TAVILY_API_KEY="YOUR_TAVILY_API_KEY_HERE"  
SERPAPI_API_KEY="YOUR_SERPAPI_API_KEY_HERE"

Add `.env` to `.gitignore` to avoid committing secrets, or simply copy and paste your API keys interactively in the terminal once you are prompted to after starting the chatbot.

## Usage
Start the chatbot:

python main.py

You will then see:

🧑 You:

## Example Interactions
- Plan a 7-day trip to Italy.  
- I need a flight from NYC to London next month.  
- Find a hotel in Tokyo for 3 nights for 2 adults.  
- What are the best attractions in Rome?  
- Hi there!

## To exit:

quit
