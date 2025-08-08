"""
LLM Integration for Enhanced Tool Calling Framework
"""

from .llm_wrapper import (
    BaseLLMWrapper,
    OpenAIWrapper,
    OllamaWrapper,
    HuggingFaceWrapper,
    LLMFactory,
    LLMResponse
)

from .gemini_wrapper import GeminiWrapper

from .prompt_manager import (
    PromptManager,
    PromptTemplate,
    ToolDescription
)

from .llm_agent import (
    LLMAgent,
    AgentConfig,
    ConversationTurn,
    AgentResponse
)

from .function_tool_adapter import (
    FunctionToolAdapter,
    FunctionRegistry,
    create_tool_from_function
)

from .root_agent import (
    LlmAgent,
    create_agent,
    load_env_file
)

__all__ = [
    # LLM Wrappers
    "BaseLLMWrapper",
    "OpenAIWrapper", 
    "OllamaWrapper",
    "HuggingFaceWrapper",
    "GeminiWrapper",
    "LLMFactory",
    "LLMResponse",
    
    # Prompt Management
    "PromptManager",
    "PromptTemplate", 
    "ToolDescription",
    
    # Agent Components
    "LLMAgent",
    "AgentConfig",
    "ConversationTurn",
    "AgentResponse",
    
    # Function Tool Adapter
    "FunctionToolAdapter",
    "FunctionRegistry",
    "create_tool_from_function",
    
    # Root Agent (Main API)
    "LlmAgent",
    "create_agent",
    "load_env_file"
]

