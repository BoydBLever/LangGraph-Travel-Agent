from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from llm_config import llm
from state import TravelPlannerState

def create_router():
    """Creates a router for the three travel agents using LangGraph patterns"""

    router_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a routing expert for a travel planning system.\n\n        Analyze the user's query and decide which specialist agent should handle it:\n\n        - FLIGHT: Flight bookings, airlines, air travel, flight search, tickets, airports, departures, arrivals, airline prices\n        - HOTEL: Hotels, accommodations, stays, rooms, hotel bookings, lodging, resorts, hotel search, hotel prices\n        - ITINERARY: Travel itineraries, trip planning, destinations, activities, attractions, sightseeing, travel advice, weather, culture, food, general travel questions\n\n        Respond with ONLY one word: FLIGHT, HOTEL, or ITINERARY\n\n        Examples:\n        "Book me a flight to Paris" → FLIGHT\n        "Find hotels in Tokyo" → HOTEL\n        "Plan my 5-day trip to Italy" → ITINERARY\n        "Search flights from NYC to London" → FLIGHT\n        "Where should I stay in Bali?" → HOTEL\n        "What are the best attractions in Rome?" → ITINERARY\n        "I need airline tickets" → FLIGHT\n        "Show me hotel options" → HOTEL\n        "Create an itinerary for Japan" → ITINERARY"""),

        ("user", "Query: {query}")
    ])

    router_chain = router_prompt | llm | StrOutputParser()
    def route_query(state: TravelPlannerState):
        """Router function for LangGraph - decides which agent to call next"""

        # Get the latest user message
        user_message = state["messages"][-1].content

        print(f"🧭 Router analyzing: '{user_message[:50]}...'\n")

        try:
            # Get LLM routing decision
            decision = router_chain.invoke({"query": user_message}).strip().upper()

            # Map to our agent node names
            agent_mapping = {
                "FLIGHT": "flight_agent",
                "HOTEL": "hotel_agent",
                "ITINERARY": "itenary_agent"
            }

            next_agent = agent_mapping.get(decision, "itenary_agent")
            print(f"🎯 Router decision: {decision} → {next_agent}")

            return next_agent

        except Exception as e:
            print(f"⚠️ Router error, defaulting to itenary_agent: {e}")
            return "itenary_agent"

    return route_query

def router_node(state: TravelPlannerState):
    """Router node - determines which agent should handle the query"""
    # Assuming create_router is called once globally or passed in
    # For simplicity, we'll create it here each time, or ensure it's a global call in main
    router_func = create_router() 
    next_agent = router_func(state)

    return {
        "next_agent": next_agent,
        "user_query": state["messages"][-1].content # User query is already in messages, but explicit for clarity
    }