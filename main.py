import os
import getpass
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# Relative import for the travel_planner graph

# --- Environment Setup (Crucial to run first) ---
load_dotenv() # Load any existing .env file

# Prompt for API keys if not already set in environment variables
# For deployment, these should be securely managed, not hardcoded or prompted interactively.
if not os.getenv("GOOGLE_API_KEY"): os.environ["GOOGLE_API_KEY"] = getpass.getpass("GOOGLE_API_KEY")
if not os.getenv("TAVILY_API_KEY"): os.environ["TAVILY_API_KEY"] = getpass.getpass("TAVILY_API_KEY")
if not os.getenv("SERPAPI_API_KEY"): os.environ["SERPAPI_API_KEY"] = getpass.getpass("SERPAPI_API_KEY")
print("Environment variables loaded.\n")
# --------------------------------------------------
from graph_builder import travel_planner

def test_system(query):
    """Test our multi-agent system"""
    print(f"🧑 User: {query}")

    # Create initial state
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "next_agent": "" # Will be set by the router
    }

    # Run the system
    result = travel_planner.invoke(initial_state)

    # Get the response
    response = result["messages"][-1].content
    print(f"\n🤖 Assistant: {response}")
    print("-" * 50)

def multi_turn_chat():
    """Multi-turn conversation with checkpoint memory"""
    print("💬 Multi-Agent Travel Assistant (Multi-turn Mode)")
    print("=" * 50)

    # For multi-turn, you need a consistent thread/session ID
    # The InMemorySaver uses this. For persistent memory, use SQLiteSaver.
    config = {"configurable": {"thread_id": "1"}}

    while True:
        user_input = input("\n🧑 You: ")

        if user_input.lower() == 'quit':
            print("Exiting chat. Goodbye!")
            break

        print(f"\n📊 Processing query...")

        # For multi-turn, just add the new message
        # The graph will maintain conversation history automatically using the checkpointer
        result = travel_planner.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config
        )

        response = result["messages"][-1].content
        print(f"\n🤖 Assistant: {response}")
        print("-" * 50)

# --- Run the chatbot ---
if __name__ == "__main__":
    # You can uncomment these for quick tests
    # test_system("I need to book a flight to Paris")
    # test_system("Find me a good hotel in New Delhi on 15 July 2025 for 1 night for 1 adult")

    multi_turn_chat()