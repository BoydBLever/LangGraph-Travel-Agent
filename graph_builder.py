from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from typing import Literal

from state import TravelPlannerState
from agents import itinerary_agent_node, flight_agent_node, hotel_agent_node
from router import router_node # Assuming router_node now creates the router internally or accesses a global one

# Conditional routing function
def route_to_agent(state: TravelPlannerState) -> Literal["flight_agent", "hotel_agent", "itenary_agent"]:
    """Conditional edge function - routes to appropriate agent based on router decision"""
    # The router_node already sets 'next_agent', so we just return it
    return state["next_agent"]

# Build the complete travel planning graph
workflow = StateGraph(TravelPlannerState)

# Add all nodes to the graph
workflow.add_node("router", router_node)
workflow.add_node("flight_agent", flight_agent_node)
workflow.add_node("hotel_agent", hotel_agent_node)
workflow.add_node("itenary_agent", itinerary_agent_node)

# Set entry point - always start with router
workflow.set_entry_point("router")

# Add conditional edge from router to appropriate agent
workflow.add_conditional_edges(
    "router",
    route_to_agent, # Use the conditional function here
    {
        "flight_agent": "flight_agent",
        "hotel_agent": "hotel_agent",
        "itenary_agent": "itenary_agent"
    }
)

# Add edges from each agent back to END
workflow.add_edge("flight_agent", END)
workflow.add_edge("hotel_agent", END)
workflow.add_edge("itenary_agent", END)

checkpointer = InMemorySaver()

# Compile the graph
travel_planner = workflow.compile(checkpointer=checkpointer)

print("✅ Travel Planning Graph built successfully!")