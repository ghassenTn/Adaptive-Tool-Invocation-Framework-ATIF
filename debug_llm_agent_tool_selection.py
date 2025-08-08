import asyncio
import os
from llm_integration.root_agent import LlmAgent
from llm_integration.gemini_wrapper import GeminiWrapper
from dotenv import load_dotenv
# --- Custom Tools ---

def get_current_user_name() -> str:
    """
    return the user name like ('john')
    """
    return 'Ghassen saidi'

def get_current_user_age() -> int:
    """
    return the user age like 26
    """
    return 26


async def run_advanced_debug():
    print("🚀 Advanced LLM Agent Tool Selection Demo")
    print("=" * 60)

    # Load environment variables
    load_dotenv()
    print()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("Error: API key missing.")
        return

    # Initialize Agent
    root_agent = LlmAgent(
        model=GeminiWrapper(model_name='gemini-2.5-flash-preview-05-20', api_key=api_key),
        name="advanced_debug_agent",
        instruction="You are an assistant with tools for return the user informations ",
        tools=[get_current_user_name, get_current_user_age],
        language="en"
    )

    print(f"Agent initialized: {root_agent}")
    print(f"Available tools: {root_agent.list_tools()}")

    # Test queries
    test_queries = [
        " whats is my name ? ",
        " whats is my age ? " 
        ]

    for query in test_queries:
        print(f"\n📌 User Query: {query}")
        print("-" * 40)
        response = await root_agent.run_async(query)

        # Show raw LLM output for debugging
        if hasattr(response, "raw_llm_output"):
            print(f"--- Raw LLM Output ---\n{response.raw_llm_output}\n----------------------")

        print(f"✅ Agent Response: {response.message}")
        if response.tool_used:
            print(f"🔧 Tool Used: {response.tool_used}")
            print(f"📊 Tool Result: {response.tool_result}")
        print(f"💡 Success: {response.success}")
        print(f"🧠 Reasoning: {response.reasoning}")
        if response.error:
            print(f"⚠️ Error: {response.error}")

if __name__ == "__main__":
    asyncio.run(run_advanced_debug())
