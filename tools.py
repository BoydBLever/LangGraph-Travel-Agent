import os
import json
import serpapi
from langchain_tavily import TavilySearch
from langchain.tools import Tool

# Tavily Search Tool
tool = TavilySearch(max_results=2)
tools = [tool] # for binding to itinerary agent

# Flight Search Tool
def search_flights(departure_airport: str, arrival_airport: str, outbound_date: str, return_date: str = None, adults: int = 1, children: int = 0) -> str:
    """
    Search for flights using Google Flights engine.

    Args:
        departure_airport: Departure airport code (e.g., 'NYC', 'LAX')
        arrival_airport: Arrival airport code (e.g., 'LON', 'NRT')
        outbound_date: Departure date (YYYY-MM-DD format)
        return_date: Return date (YYYY-MM-DD format, optional for one-way)
        adults: Number of adult passengers (default: 1)
        children: Number of child passengers (default: 0)
    """
    params = {
        'api_key': os.environ.get('SERPAPI_API_KEY'),
        'engine': 'google_flights',
        'hl': 'en',
        'gl': 'us',
        'departure_id': departure_airport,
        'arrival_id': arrival_airport,
        'outbound_date': outbound_date,
        'return_date': return_date,
        'currency': 'USD',
        'adults': adults,
        'children': children,
        'stops': '1'
    }

    try:
        search = serpapi.search(params)
        results = search.data.get('best_flights', [])
        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Flight search failed: {str(e)}"

# Define the flight search tool explicitly for binding
search_flights_tool = Tool.from_function(
    func=search_flights,
    name="search_flights",
    description="Search for flights using Google Flights engine."
)

# Hotel Search Tool
def search_hotels(location: str, check_in_date: str, check_out_date: str, adults: int = 1, children: int = 0, rooms: int = 1, hotel_class: str = None, sort_by: int = 8) -> str:
    """
    Search for hotels using Google Hotels engine.

    Args:
        location: Location to search for hotels (e.g., 'New York', 'Paris', 'Tokyo')
        check_in_date: Check-in date (YYYY-MM-DD format)
        check_out_date: Check-out date (YYYY-MM-DD format)
        adults: Number of adults (default: 1)
        children: Number of children (default: 0)
        rooms: Number of rooms (default: 1)
        hotel_class: Hotel class filter (e.g., '2,3,4' for 2-4 star hotels)
        sort_by: Sort parameter (default: 8 for highest rating)
    """

    # Ensure proper integer types
    adults = int(float(adults)) if adults else 1
    children = int(float(children)) if children else 0
    rooms = int(float(rooms)) if rooms else 1
    sort_by = int(float(sort_by)) if sort_by else 8

    params = {
        'api_key': os.environ.get('SERPAPI_API_KEY'),
        'engine': 'google_hotels',
        'hl': 'en',
        'gl': 'us',
        'q': location,
        'check_in_date': check_in_date,
        'check_out_date': check_out_date,
        'currency': 'USD',
        'adults': adults,
        'children': children,
        'rooms': rooms,
        'sort_by': sort_by
    }

    # Only add hotel_class if provided
    if hotel_class:
        params['hotel_class'] = hotel_class

    try:
        search = serpapi.search(params)
        properties = search.data.get('properties', [])

        if not properties:
            return f"No hotels found. Available data keys: {list(search.data.keys())}"

        # Return top 5 results
        return json.dumps(properties[:5], indent=2)

    except Exception as e:
        return f"Hotel search failed: {str(e)}"

# Define the hotel search tool explicitly for binding
search_hotels_tool = Tool.from_function(
    func=search_hotels,
    name="search_hotels",
    description="Search for hotels using Google Hotels engine."
)
