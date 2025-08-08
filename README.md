# Adaptive Tool Invocation Framework (ATIF)

ATIF is a framework designed to enhance the tool-calling capabilities of Large Language Models (LLMs), with a focus on improving the performance of less powerful models.

## Overview

ATIF addresses the challenge of inaccurate tool invocation in less powerful LLMs by providing intelligent layers that assist with:

-   **Smart Tool Selection:** Advanced semantic matching and enhanced keyword matching to select the most appropriate tool.
-   **Sophisticated Argument Extraction:** Multiple strategies for extracting arguments from natural language, with support for various data types and automatic validation.
-   **Reliable Execution:** Smart retries, parallel execution, and comprehensive error handling.
-   **Adaptive Learning:** A comprehensive feedback system, detailed performance metrics, and continuous improvement.

## LLM Integration

ATIF is designed to be flexible, allowing you to integrate any LLM of your choice, whether it's a cloud-based model like Google Gemini or a local model. The framework enhances the LLM's ability to call tools intelligently by providing specialized auxiliary layers.

### Using LlmAgent with Gemini

You can use the `LlmAgent` as a master agent that interacts with the Gemini model (or any other LLM) and uses the tools you define.

**1. Set up the `.env` file:**

Create a file named `.env` in the root of your project and add your API key:

```
GEMINI_API_KEY=YOUR_GEMINI_API_KEY_HERE
```

**2. Define the tools:**

You can define your tools as regular Python functions and then pass them to the `LlmAgent`. The framework will automatically convert them into a format that the LLM can understand.

```python
def get_weather(cityName: str):
    """
    Return the current weather in a specific city.
    Args:
        cityName (str): The city name.
    Returns:
        The current weather of a specific city.
    """
    return f"The current weather in {cityName} is 35 degrees Celsius and sunny."
```

**3. Initialize and run the LlmAgent:**

```python
import asyncio
import os
from llm_integration.root_agent import LlmAgent, load_env_file
from llm_integration.gemini_wrapper import GeminiWrapper

# Load environment variables from the .env file
load_env_file()
api_key = os.getenv("GOOGLE_AI_API_KEY")

if not api_key:
    print("Error: GOOGLE_AI_API_KEY not found in .env file. Please provide it.")
else:
    # Initialize the Gemini LLM model
    model = GeminiWrapper(model_name="gemini-1.5-flash-latest", api_key=api_key)

    # Initialize the master agent
    root_agent = LlmAgent(
        model=model,
        name='weather_provider',
        instruction=(
            "You are a helpful weather assistant. Use the available tools to answer user questions about the weather."
        ),
        tools=[get_weather],
        language="en"
    )

    async def run_agent_example():
        print(f"Agent initialized: {root_agent}")
        print(f"Available tools: {root_agent.list_tools()}")

        # Test queries
        test_queries = [
            "What is the weather in Sfax, Tunisia?",
            "Can you tell me the weather in London?",
            "Hello, how are you?"
        ]

        for query in test_queries:
            print(f"\nUser Query: {query}")
            print("-" * 30)
            response = await root_agent.run_async(query)
            print(f"Agent Response: {response.message}")
            if response.tool_used:
                print(f"  Tool Used: {response.tool_used}")
                print(f"  Tool Result: {response.tool_result}")
            print(f"  Success: {response.success}")
            print(f"  Reasoning: {response.reasoning}")
            if response.error:
                print(f"  Error: {response.error}")

    if __name__ == "__main__":
        asyncio.run(run_agent_example())
```

With this setup, the `LlmAgent` can interact with Gemini, provide it with a description of the tools, and then analyze Gemini's response to determine the appropriate tool, extract the parameters, and execute it, providing an intelligent tool-calling experience.