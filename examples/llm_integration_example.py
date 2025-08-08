"""
Example of integrating an LLM with the Enhanced Tool Calling Framework.

This example demonstrates how to use the LlmAgent with a Gemini model
and custom tools.
"""

import asyncio
import os
from typing import Dict, Any, List, Optional

# Import the new LlmAgent and related components
from enhanced_tool_calling_framework.llm_integration.root_agent import LlmAgent, load_env_file
from enhanced_tool_calling_framework.llm_integration.gemini_wrapper import GeminiWrapper

# --- Define a custom tool --- 
# This function will be automatically converted into a tool by FunctionToolAdapter
def get_weather(cityName: str):
    """
    return the current weather in a specific city
    Args:
        cityName(str): the city name
    Return:
        the current weather of a specific city 
    """
    # In a real application, you would call a weather API here
    # For this example, we'll return a mock response
    return f"The current weather in {cityName} is 35 degrees Celsius and sunny."


async def main():
    # 1. Load environment variables (e.g., GOOGLE_AI_API_KEY)
    # Make sure you have a .env file in the root of the project
    # with GOOGLE_AI_API_KEY=YOUR_API_KEY_HERE
    env_vars = load_env_file("/home/ubuntu/enhanced_tool_calling_framework/.env")
    
    # Check if API key is available
    if "GOOGLE_AI_API_KEY" not in os.environ:
        print("Error: GOOGLE_AI_API_KEY not found in .env file or environment variables.")
        print("Please add GOOGLE_AI_API_KEY=YOUR_API_KEY_HERE to /home/ubuntu/enhanced_tool_calling_framework/.env")
        return

    print("Initializing LlmAgent with Gemini model...")
    
    # 2. Initialize the LlmAgent
    # The model parameter can be a string (e.g., "gemini-1.5-flash")
    # or an instance of a BaseLLMWrapper (e.g., GeminiWrapper())
    root_agent = LlmAgent(
        model="gemini-1.5-flash",  # Or "gemini-1.5-pro" for more advanced capabilities
        name='weather_agent',
        instruction=(
            "You are a helpful assistant that can provide weather information. "
            "Use the available tools to answer user questions about the weather."
        ),
        tools=[get_weather],
        language="ar" # Set language to Arabic as requested
    )
    
    print(f"Agent initialized: {root_agent}")
    print(f"Available tools: {root_agent.list_tools()}")
    print(f"Model info: {root_agent.get_model_info()}")

    # 3. Interact with the agent
    print("\n--- Agent Interaction ---")
    
    user_query_1 = "ما هو الطقس في صفاقس، تونس؟"
    print(f"User: {user_query_1}")
    response_1 = await root_agent.run_async(user_query_1)
    print(f"Agent: {response_1.message}")
    if response_1.tool_used:
        print(f"  (Tool Used: {response_1.tool_used}, Result: {response_1.tool_result})")

    print("\n--- Another Interaction ---")
    user_query_2 = "هل يمكنك أن تخبرني عن الطقس في لندن؟"
    print(f"User: {user_query_2}")
    response_2 = await root_agent.run_async(user_query_2)
    print(f"Agent: {response_2.message}")
    if response_2.tool_used:
        print(f"  (Tool Used: {response_2.tool_used}, Result: {response_2.tool_result})")

    print("\n--- Interaction without tool --- ")
    user_query_3 = "مرحباً، كيف حالك؟"
    print(f"User: {user_query_3}")
    response_3 = await root_agent.run_async(user_query_3)
    print(f"Agent: {response_3.message}")

    print("\n--- Conversation Summary ---")
    print(root_agent.get_conversation_summary())

if __name__ == "__main__":
    asyncio.run(main())

