from langchain.agents import tool
from tools import query_medgemma, call_emergency

@tool
def ask_mental_health_specialist(query: str) -> str:
    """
    Generate a therapeutic response using the MedGemma model.
    Use this for all general user queries, mental health questions, emotional concerns,
    or to offer empathetic, evidence-based guidance in a conversational tone.
    """
    return query_medgemma(query)


@tool
def emergency_call_tool() -> None:
    """
    Place an emergency call to the safety helpline's phone number via Twilio.
    Use this only if the user expresses suicidal ideation, intent to self-harm,
    or describes a mental health emergency requiring immediate help.
    """
    call_emergency()


import requests
# from config import LOCATIONIQ_API_KEY
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LOCATIONIQ_API_KEY

@tool
def find_nearby_therapists_by_location(location: str) -> str:
    """
    Finds nearby therapists/medical facilities using LocationIQ API.

    Args:
        location (str): The city or area to search.

    Returns:
        str: A list of nearby places with names and addresses.
    """
    try:
        # Step 1: Geocode location
        geocode_url = "https://us1.locationiq.com/v1/search"

        geocode_params = {
            "key": LOCATIONIQ_API_KEY,
            "q": location,
            "format": "json",
            "limit": 1
        }

        geocode_response = requests.get(
            geocode_url,
            params=geocode_params,
            timeout=10
        )

        geocode_response.raise_for_status()
        geocode_res = geocode_response.json()

        if not geocode_res:
            return f"Location '{location}' not found."

        lat = geocode_res[0]["lat"]
        lon = geocode_res[0]["lon"]

        # Step 2: Search nearby doctors/clinics
        nearby_url = "https://us1.locationiq.com/v1/nearby"

        nearby_params = {
            "key": LOCATIONIQ_API_KEY,
            "lat": lat,
            "lon": lon,
            "tag": "amenity:doctors",
            "radius": 5000,
            "format": "json",
            "limit": 5
        }

        nearby_response = requests.get(
            nearby_url,
            params=nearby_params,
            timeout=10
        )

        nearby_response.raise_for_status()
        places = nearby_response.json()

        if not places:
            return f"No therapists/clinics found near {location}."

        output = [f"Nearby therapists/clinics near {location}:"]

        for place in places:
            name = place.get("name") or "Unknown"
            address = place.get(
                "display_name",
                "Address not available"
            )

            # LocationIQ nearby response may not contain phone
            phone = place.get(
                "phone",
                "Phone not available"
            )

            output.append(
                f"- {name} | {address} | Phone: {phone}"
            )

        return "\n".join(output)

    except Exception as e:
        return f"Error finding therapists: {str(e)}"
# @tool
# def find_nearby_therapists_by_location(location: str) -> str:
#     """
#     Finds real therapists near the specified location using LocationIQ API.
    
#     Args:
#         location (str): The city or area to search.
    
#     Returns:
#         str: A list of therapist names, addresses, and phone numbers.
#     """
#     try:
#         # Step 1: Geocode the location
#         geocode_url = "https://us1.locationiq.com/v1/search"
#         geocode_params = {
#             "key": LOCATIONIQ_API_KEY,
#             "q": location,
#             "format": "json",
#             "limit": 1
#         }
#         geocode_res = requests.get(geocode_url, params=geocode_params).json()
        
#         if not geocode_res:
#             return f"Location '{location}' not found."

#         lat = geocode_res[0]["lat"]
#         lon = geocode_res[0]["lon"]

#         # Step 2: Search nearby therapists
#         nearby_url = "https://us1.locationiq.com/v1/nearby"
#         nearby_params = {
#             "key": LOCATIONIQ_API_KEY,
#             "lat": lat,
#             "lon": lon,
#             "tag": "amenity:doctors",   # doctors / clinic
#             "radius": 5000,
#             "format": "json",
#             "limit": 5
#         }
#         places = requests.get(nearby_url, params=nearby_params).json()

#         if not places:
#             return f"No therapists found near {location}."

#         output = [f"Therapists near {location}:"]
#         for place in places:
#             name = place.get("name") or place.get("display_name", "Unknown")
#             address = place.get("display_name", "Address not available")
#             output.append(f"- {name} | {address}")
#             phone=details.get("result",{}).get("formatted_phone_number","phone not available")

#         return "\n".join(output)

#     except Exception as e:
#         return f"Error finding therapists: {str(e)}"


# import googlemaps
# from config import GOOGLE_MAPS_API_KEY
# gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)


# @tool
# def find_nearby_therapists_by_location(location: str) -> str:
#     """
#     Finds real therapists near the specified location using Google Maps API.
    
#     Args:
#         location (str): The city or area to search.
    
#     Returns:
#         str: A list of therapist names, addresses, and phone numbers.
#     """
#     geocode_result = gmaps.geocode(location)
#     lat_lng = geocode_result[0]['geometry']['location']
#     lat, lng = lat_lng['lat'], lat_lng['lng']
#     places_result = gmaps.places_nearby(
#             location=(lat, lng),
#             radius=5000,
#             keyword="Psychotherapist"
#         )
#     output = [f"Therapists near {location}:"]
#     top_results = places_result['results'][:5]
#     for place in top_results:
#             name = place.get("name", "Unknown")
#             address = place.get("vicinity", "Address not available")
#             details = gmaps.place(place_id=place["place_id"], fields=["formatted_phone_number"])
#             phone = details.get("result", {}).get("formatted_phone_number", "Phone not available")

#             output.append(f"- {name} | {address} | {phone}")

    
#     return "\n".join(output)


# Step1: Create an AI Agent & Link to backend
#from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from config import GROQ_API_KEY

tools = [ask_mental_health_specialist, emergency_call_tool, find_nearby_therapists_by_location]
#llm = ChatOpenAI(model="gpt-4", temperature=0.2, api_key=OPENAI_API_KEY)
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.2, api_key=GROQ_API_KEY)
graph = create_react_agent(llm, tools=tools)

SYSTEM_PROMPT = """
You are an AI engine supporting mental health conversations with warmth and vigilance.
You have access to three tools:

1. `ask_mental_health_specialist`: Use this tool to answer all emotional or psychological queries with therapeutic guidance.
2. `find_nearby_therapists_by_location`: Use this tool if the user asks about nearby therapists or if recommending local professional help would be beneficial.
3. `emergency_call_tool`: Use this immediately if the user expresses suicidal thoughts, self-harm intentions, or is in crisis.

Always take necessary action. Respond kindly, clearly, and supportively.
"""

def parse_response(stream):
    tool_called_name = "None"
    final_response = None

    for s in stream:
        # Check if a tool was called
        tool_data = s.get('tools')
        if tool_data:
            tool_messages = tool_data.get('messages')
            if tool_messages and isinstance(tool_messages, list):
                for msg in tool_messages:
                    tool_called_name = getattr(msg, 'name', 'None')

        # Check if agent returned a message
        agent_data = s.get('agent')
        if agent_data:
            messages = agent_data.get('messages')
            if messages and isinstance(messages, list):
                for msg in messages:
                    if msg.content:
                        final_response = msg.content

    return tool_called_name, final_response


"""if __name__ == "__main__":
    while True:
        user_input = input("User: ")
        print(f"Received user input: {user_input[:200]}...")
        inputs = {"messages": [("system", SYSTEM_PROMPT), ("user", user_input)]}
        stream = graph.stream(inputs, stream_mode="updates")
        tool_called_name, final_response = parse_response(stream)
        print("TOOL CALLED: ", tool_called_name)
        print("ANSWER: ", final_response)"""
        