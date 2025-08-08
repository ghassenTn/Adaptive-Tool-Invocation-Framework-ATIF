"""
Root Agent - Main agent class that integrates LLM with tool calling framework
"""

import os
import asyncio
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass

from .gemini_wrapper import GeminiWrapper
from .llm_wrapper import BaseLLMWrapper, LLMFactory
from .llm_agent import LLMAgent, AgentConfig, AgentResponse
from .function_tool_adapter import FunctionRegistry, create_tool_from_function
from core.simple_unified_api import SimpleUnifiedToolAPI


def load_env_file(env_path: str = ".env") -> Dict[str, str]:
    """
    Load environment variables from .env file
    
    Args:
        env_path: Path to .env file
        
    Returns:
        Dictionary of environment variables
    """
    env_vars = {}
    
    if os.path.exists(env_path):
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"\'')  # Remove quotes
                    env_vars[key] = value
                    # Also set in os.environ
                    os.environ[key] = value
    
    return env_vars


class LlmAgent:
    """
    Main agent class that provides a simple interface for LLM + tool integration
    
    This class follows the user's requested API:
    root_agent = LlmAgent(
        model=model,
        name='weather_provider',
        instruction="give me the current weather in sfax, tunisia",
        tools=[get_weather]
    )
    """
    
    def __init__(
        self,
        model: Optional[Union[str, BaseLLMWrapper]] = None,
        name: str = "assistant",
        instruction: str = "",
        tools: Optional[List[Callable]] = None,
        language: str = "en",
        **kwargs
    ):
        """
        Initialize LLM Agent
        
        Args:
            model: Model name (str) or LLM wrapper instance
            name: Agent name
            instruction: Initial instruction/prompt for the agent
            tools: List of Python functions to use as tools
            language: Language for prompts ("en" or "ar")
            **kwargs: Additional configuration
        """
        self.name = name
        self.instruction = instruction
        self.language = language
        
        # Load .env file automatically
        self.env_vars = load_env_file()
        
        # Initialize LLM wrapper
        self.llm_wrapper = self._initialize_llm(model, **kwargs)
        
        # Initialize function registry and register tools
        self.function_registry = FunctionRegistry()
        if tools:
            self.function_registry.register_functions(tools)
        
        # Initialize tool API
        self.tool_api = SimpleUnifiedToolAPI()
        self.tool_api.register_tools(self.function_registry.get_all_tools())
        
        # Initialize agent config
        agent_config = AgentConfig(
            language=language,
            enable_conversation_memory=kwargs.get("enable_memory", True),
            enable_tool_feedback=kwargs.get("enable_feedback", True),
            llm_temperature=kwargs.get("temperature", 0.7),
            max_retries=kwargs.get("max_retries", 3)
        )
        
        # Initialize LLM agent
        self.agent = LLMAgent(
            llm_wrapper=self.llm_wrapper,
            tool_api=self.tool_api,
            config=agent_config
        )
        
        # Store initial instruction as system message
        self.system_instruction = instruction
    
    def _initialize_llm(self, model: Optional[Union[str, BaseLLMWrapper]], **kwargs) -> BaseLLMWrapper:
        """Initialize LLM wrapper based on model parameter"""
        
        if isinstance(model, BaseLLMWrapper):
            # Already a wrapper instance
            return model
        
        if isinstance(model, str):
            # Model name provided
            if model.startswith("gemini"):
                return GeminiWrapper(model_name=model, **kwargs)
            elif model.startswith("gpt"):
                return LLMFactory.create_llm("openai", model, **kwargs)
            elif "llama" in model.lower() or "mistral" in model.lower():
                return LLMFactory.create_llm("ollama", model, **kwargs)
            else:
                # Default to Gemini
                return GeminiWrapper(model_name=model, **kwargs)
        
        # No model specified, try to auto-detect from environment
        if "GOOGLE_AI_API_KEY" in os.environ:
            return GeminiWrapper(**kwargs)
        elif "OPENAI_API_KEY" in os.environ:
            return LLMFactory.create_llm("openai", "gpt-3.5-turbo", **kwargs)
        else:
            # Default to Gemini with flash model
            return GeminiWrapper(model_name="gemini-1.5-flash", **kwargs)
    
    async def chat(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Chat with the agent
        
        Args:
            message: User message
            context: Optional context
            
        Returns:
            Agent response as string
        """
        # Combine system instruction with user message if this is the first interaction
        if self.system_instruction and not self.agent.conversation_history:
            combined_message = f"{self.system_instruction}\n\nUser: {message}"
        else:
            combined_message = message
        
        response = await self.agent.chat(combined_message, context)
        return response.message
    
    def run(self, message: Optional[str] = None) -> str:
        """
        Synchronous interface to run the agent
        
        Args:
            message: Optional message (uses instruction if not provided)
            
        Returns:
            Agent response as string
        """
        if message is None:
            message = self.instruction
        
        if not message:
            raise ValueError("No message or instruction provided")
        
        # Run async chat in sync context
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.chat(message))
    
    async def run_async(self, message: Optional[str] = None) -> AgentResponse:
        """
        Async interface to run the agent with full response details
        
        Args:
            message: Optional message (uses instruction if not provided)
            
        Returns:
            Full AgentResponse object
        """
        if message is None:
            message = self.instruction
        
        if not message:
            raise ValueError("No message or instruction provided")
        
        # Combine system instruction with user message if this is the first interaction
        if self.system_instruction and not self.agent.conversation_history:
            combined_message = f"{self.system_instruction}\n\nUser: {message}"
        else:
            combined_message = message
        
        return await self.agent.chat(combined_message)
    
    def add_tool(self, func: Callable, name: Optional[str] = None, description: Optional[str] = None):
        """
        Add a new tool function
        
        Args:
            func: Function to add as tool
            name: Optional custom name
            description: Optional custom description
        """
        tool = self.function_registry.register_function(func, name, description)
        self.tool_api.register_tool(tool)
        
        # Refresh agent's tool cache
        self.agent.refresh_tools()
    
    def add_tools(self, functions: List[Callable]):
        """Add multiple tool functions"""
        tools = self.function_registry.register_functions(functions)
        self.tool_api.register_tools(tools)
        
        # Refresh agent's tool cache
        self.agent.refresh_tools()
    
    def remove_tool(self, name: str) -> bool:
        """
        Remove a tool by name
        
        Args:
            name: Tool name to remove
            
        Returns:
            True if removed, False if not found
        """
        if self.function_registry.unregister(name):
            # Recreate tool API with remaining tools
            self.tool_api = SimpleUnifiedToolAPI()
            self.tool_api.register_tools(self.function_registry.get_all_tools())
            
            # Update agent
            self.agent.tool_api = self.tool_api
            self.agent.refresh_tools()
            return True
        return False
    
    def list_tools(self) -> List[str]:
        """List all available tool names"""
        return self.function_registry.list_functions()
    
    def get_tool_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific tool"""
        tool = self.function_registry.get_tool(name)
        if tool:
            return {
                "name": tool.name,
                "description": tool.description,
                "parameters": [
                    {
                        "name": param.name,
                        "type": param.type.value,
                        "description": param.description,
                        "required": param.required,
                        "default": param.default
                    }
                    for param in tool.parameters
                ]
            }
        return None
    
    def clear_conversation(self):
        """Clear conversation history"""
        self.agent.clear_conversation()
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get conversation summary"""
        return self.agent.get_conversation_summary()
    
    def update_config(self, **kwargs):
        """Update agent configuration"""
        self.agent.update_config(**kwargs)
    
    def set_language(self, language: str):
        """Set agent language"""
        self.language = language
        self.agent.update_config(language=language)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return self.llm_wrapper.get_model_info()
    
    def __repr__(self) -> str:
        return f"LlmAgent(name='{self.name}', model='{self.llm_wrapper.model_name}', tools={len(self.function_registry.tools)})"


# Convenience function for creating agents
def create_agent(
    model: Optional[Union[str, BaseLLMWrapper]] = None,
    name: str = "assistant",
    instruction: str = "",
    tools: Optional[List[Callable]] = None,
    **kwargs
) -> LlmAgent:
    """
    Convenience function to create an LLM agent
    
    Args:
        model: Model name or wrapper instance
        name: Agent name
        instruction: System instruction
        tools: List of tool functions
        **kwargs: Additional configuration
        
    Returns:
        LlmAgent instance
    """
    return LlmAgent(
        model=model,
        name=name,
        instruction=instruction,
        tools=tools,
        **kwargs
    )

