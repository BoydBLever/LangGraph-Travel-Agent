from langchain_core.messages import ToolMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import json

from state import TravelState, TravelPlannerState
from tools import tool, search_flights, search_flights_tool, search_hotels, search_hotels_tool
from llm_config import llm

# --- Warm-up Agents (Optional - not used in final multi-agent graph) ---
# If you don't need these, you can omit them
def destination_agent(state: TravelState) -> TravelState:
    print("🔧 destination_agent running…")
    q = state.get("user_input", "").lower()

    if "beach" in q:
        dest = "Bali"
    elif "snow" in q or "mountain" in q:
        dest = "Manali"
    else:
        dest = "Kyoto"  # sensible default

    print(f"🌍 Suggested destination: {dest}")
    return {**state, "destination": dest}

def itinerary_agent_warmup(state: TravelState) -> TravelState:
    print("🔧 itinerary_agent running…")
    dest = state["destination"]
    plan = (
        f"Day 1: Arrive in {dest}\n"
        f"Day 2: Explore iconic spots in {dest}\n"
        f"Day 3: Relax + sample local cuisine"
    )
    print(f"🧳 Draft itinerary:\n{plan}")
    return {**state, "itinerary": plan}

def activity_agent_warmup(state: TravelState) -> TravelState:
    print("🔧 activity_agent running…")
    dest = state["destination"]
    act = (
        "Snorkelling • Beach yoga"
        if dest == "Bali"
        else "Skiing • Mountain trek"
    )
    print(f"🎯 Suggested activities: {act}")
    return {**state, "activities": act}
# ----------------------------------------------------------------------

# --- Itinerary Agent --- #
itenary_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert travel itinerary planner. ONLY respond to travel planning and itinerary-related questions.\n\nIMPORTANT RULES:\n- If asked about non-travel topics (weather, math, general questions), politely decline and redirect to travel planning\n- Always provide complete, well-formatted itineraries with specific details\n- Include timing, locations, transportation, and practical tips\n\nUse the ReAct approach:\n1. THOUGHT: Analyze what travel information is needed\n2. ACTION: Search for current information about destinations, attractions, prices, hours\n3. OBSERVATION: Process the search results\n4. Provide a comprehensive, formatted response\n\nAvailable tools:\n- TavilySearch: Search for current travel information\n\nFormat your itineraries with:\n- Clear day-by-day breakdown\n- Specific times and locations\n- Transportation between locations\n- Estimated costs when possible\n- Practical tips and recommendations"""),
    MessagesPlaceholder(variable_name="messages"),
])

llm_with_tavily_tools = llm.bind_tools([tool]) # bind the TavilySearch tool
itenary_agent = itenary_prompt | llm_with_tavily_tools

def itinerary_agent_node(state: TravelPlannerState):
    """Itinerary planning agent node"""
    messages = state["messages"]
    response = itenary_agent.invoke({"messages": messages})

    # Handle tool calls if present
    if hasattr(response, 'tool_calls') and response.tool_calls:
        tool_messages = []
        for tool_call in response.tool_calls:
            if tool_call['name'] == 'tavily_search_results_json':
                try:
                    # Directly invoke the TavilySearch tool from tools.py
                    tool_result = tool.search(query=tool_call['args']['query'], max_results=2)
                    tool_result = json.dumps(tool_result, indent=2)
                except Exception as e:
                    tool_result = f"Search failed: {str(e)}"

                tool_messages.append(ToolMessage(
                    content=tool_result,
                    tool_call_id=tool_call['id']
                ))

        if tool_messages:
            all_messages = messages + [response] + tool_messages
            final_response = itenary_agent.invoke({"messages": all_messages})
            return {"messages": [response] + tool_messages + [final_response]}

    return {"messages": [response]}

# --- Flight Agent --- #
flight_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a flight booking expert. ONLY respond to flight-related queries.\n\nIMPORTANT RULES:\n- If asked about non-flight topics, politely decline and redirect to flight booking\n- Always use the search_flights tool to find current flight information\n- You CAN search for flights and analyze the results for:\n  * Direct flights vs connecting flights\n  * Different airlines and flight classes\n  * Various price ranges and timing options\n  * Flight duration and layover information\n- When users ask for specific preferences (direct flights, specific class, etc.), search first then filter/analyze the results\n- Present results clearly organized by outbound and return flights\n\nAvailable tools:\n- search_flights: Search for comprehensive flight data that includes all airlines, classes, and connection types\n\nProcess:\n1. ALWAYS search for flights first using the tool\n2. Analyze the results to find flights matching user preferences\n3. Present organized results with clear recommendations\n\nAirport code mapping:\n- Delhi: DEL\n- London Heathrow: LHR\n- New York: JFK/LGA/EWR\n- etc."""),
    MessagesPlaceholder(variable_name="messages"),
])

llm_with_flight_tools = llm.bind_tools([search_flights_tool])
flight_agent = flight_prompt | llm_with_flight_tools

def flight_agent_node(state: TravelPlannerState):
    """Flight booking agent node"""
    messages = state["messages"]
    response = flight_agent.invoke({"messages": messages})

    if hasattr(response, 'tool_calls') and response.tool_calls:
        tool_messages = []
        for tool_call in response.tool_calls:
            if tool_call['name'] == 'search_flights':
                try:
                    # Directly invoke the search_flights function from tools.py
                    tool_result = search_flights(**tool_call['args'])
                except Exception as e:
                    tool_result = f"Flight search failed: {str(e)}"

                tool_messages.append(ToolMessage(
                    content=tool_result,
                    tool_call_id=tool_call['id']
                ))

        if tool_messages:
            all_messages = messages + [response] + tool_messages
            final_response = flight_agent.invoke({"messages": all_messages})
            return {"messages": [response] + tool_messages + [final_response]}

    return {"messages": [response]}

# --- Hotel Agent --- #
hotel_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a hotel booking expert. ONLY respond to hotel and accommodation-related queries.\n\nIMPORTANT RULES:\n- If asked about non-hotel topics, politely decline and redirect to hotel booking\n- Always use the search_hotels tool to find current hotel information\n- Provide detailed hotel options with prices, ratings, amenities, and location details\n- Include practical booking advice and tips\n- You CAN search and analyze results for different criteria like star ratings, price ranges, amenities\n\nAvailable tools:\n- search_hotels: Search for hotels using Google Hotels engine\n\nWhen searching hotels, extract or ask for:\n- Location/destination\n- Check-in and check-out dates (YYYY-MM-DD format)\n- Number of guests (adults, children)\n- Number of rooms\n- Hotel preferences (star rating, amenities, etc.)\n\nPresent results with:\n- Hotel name and star rating\n- Price per night and total cost\n- Key amenities and features\n- Location and nearby attractions\n- Booking recommendations"""),
    MessagesPlaceholder(variable_name="messages"),
])

llm_with_hotel_tools = llm.bind_tools([search_hotels_tool])
hotel_agent = hotel_prompt | llm_with_hotel_tools

def hotel_agent_node(state: TravelPlannerState):
    """Hotel booking agent node"""
    messages = state["messages"]
    response = hotel_agent.invoke({"messages": messages})

    if hasattr(response, 'tool_calls') and response.tool_calls:
        tool_messages = []
        for tool_call in response.tool_calls:
            if tool_call['name'] == 'search_hotels':
                try:
                    # Directly invoke the search_hotels function from tools.py
                    tool_result = search_hotels(**tool_call['args'])
                except Exception as e:
                    tool_result = f"Hotel search failed: {str(e)}"

                tool_messages.append(ToolMessage(
                    content=tool_result,
                    tool_call_id=tool_call['id']
                ))

        if tool_messages:
            all_messages = messages + [response] + tool_messages
            final_response = hotel_agent.invoke({"messages": all_messages})
            return {"messages": [response] + tool_messages + [final_response]}

    return {"messages": [response]}
