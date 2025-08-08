"""
Enhanced Tool Calling Framework - Core Components
"""

# Base components
from .base_tool import BaseTool, ToolParameter, ParameterType, ToolResult

# Simple components (no external dependencies)
from .simple_tool_selector import SimpleToolSelector, ToolSelectionResult
from .simple_argument_generator import SimpleArgumentGenerator, ArgumentExtractionResult
from .simple_unified_api import SimpleUnifiedToolAPI, ToolCallRequest, ToolCallResponse

# Advanced components
from .execution_layer import (
    EnhancedToolExecutionLayer, 
    ExecutionResult, 
    ExecutionConfig, 
    RetryStrategy
)
from .feedback_loop import FeedbackLoop, FeedbackType, PerformanceMetrics

# Try to import advanced components with external dependencies
try:
    from .tool_selector import IntelligentToolSelector
    ADVANCED_TOOL_SELECTOR_AVAILABLE = True
except ImportError:
    ADVANCED_TOOL_SELECTOR_AVAILABLE = False

try:
    from .argument_generator import ToolArgumentGenerator
    ADVANCED_ARGUMENT_GENERATOR_AVAILABLE = True
except ImportError:
    ADVANCED_ARGUMENT_GENERATOR_AVAILABLE = False

try:
    from .unified_api import UnifiedToolAPI
    ADVANCED_UNIFIED_API_AVAILABLE = True
except ImportError:
    ADVANCED_UNIFIED_API_AVAILABLE = False

__version__ = "1.0.0"

__all__ = [
    # Base components
    "BaseTool",
    "ToolParameter", 
    "ParameterType",
    "ToolResult",
    
    # Simple components (always available)
    "SimpleToolSelector",
    "SimpleArgumentGenerator", 
    "SimpleUnifiedToolAPI",
    "ToolSelectionResult",
    "ArgumentExtractionResult",
    "ToolCallRequest",
    "ToolCallResponse",
    
    # Execution and feedback
    "EnhancedToolExecutionLayer",
    "ExecutionResult",
    "ExecutionConfig",
    "RetryStrategy",
    "FeedbackLoop",
    "FeedbackType",
    "PerformanceMetrics",
    
    # Feature flags
    "ADVANCED_TOOL_SELECTOR_AVAILABLE",
    "ADVANCED_ARGUMENT_GENERATOR_AVAILABLE", 
    "ADVANCED_UNIFIED_API_AVAILABLE"
]

# Conditional exports for advanced components
if ADVANCED_TOOL_SELECTOR_AVAILABLE:
    __all__.append("IntelligentToolSelector")

if ADVANCED_ARGUMENT_GENERATOR_AVAILABLE:
    __all__.append("ToolArgumentGenerator")

if ADVANCED_UNIFIED_API_AVAILABLE:
    __all__.append("UnifiedToolAPI")

