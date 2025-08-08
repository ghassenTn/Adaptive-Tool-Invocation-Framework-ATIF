"""
Function Tool Adapter - Converts regular Python functions to framework-compatible tools
"""

import inspect
import json
import asyncio
from typing import Dict, Any, List, Optional, Callable, Union, get_type_hints
from dataclasses import dataclass

from core.base_tool import BaseTool, ToolParameter, ParameterType, ToolResult


class FunctionToolAdapter(BaseTool):
    """
    Adapter that converts a regular Python function into a framework-compatible tool
    """
    
    def __init__(self, func: Callable, name: Optional[str] = None, description: Optional[str] = None):
        """
        Initialize function tool adapter
        
        Args:
            func: Python function to wrap
            name: Optional custom name (defaults to function name)
            description: Optional custom description (defaults to function docstring)
        """
        self.func = func
        self.is_async = asyncio.iscoroutinefunction(func)
        
        # Extract function metadata
        func_name = name or func.__name__
        func_description = description or self._extract_description(func)
        parameters = self._extract_parameters(func)
        
        super().__init__(
            name=func_name,
            description=func_description,
            parameters=parameters
        )
    
    def _extract_description(self, func: Callable) -> str:
        """Extract description from function docstring"""
        docstring = inspect.getdoc(func)
        if docstring:
            # Take the first line or paragraph as description
            lines = docstring.strip().split('\n')
            description = lines[0].strip()
            
            # If first line is empty, try to find the first non-empty line
            if not description and len(lines) > 1:
                for line in lines[1:]:
                    line = line.strip()
                    if line and not line.startswith(('Args:', 'Arguments:', 'Parameters:', 'Returns:', 'Return:')):
                        description = line
                        break
            
            return description
        
        return f"Function {func.__name__}"
    
    def _extract_parameters(self, func: Callable) -> List[ToolParameter]:
        """Extract parameters from function signature and docstring"""
        parameters = []
        
        # Get function signature
        sig = inspect.signature(func)
        
        # Get type hints
        try:
            type_hints = get_type_hints(func)
        except:
            type_hints = {}
        
        # Parse docstring for parameter descriptions
        param_descriptions = self._parse_docstring_parameters(func)
        
        for param_name, param in sig.parameters.items():
            # Skip *args and **kwargs
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            
            # Determine parameter type
            param_type = self._get_parameter_type(param, type_hints.get(param_name))
            
            # Check if required (no default value)
            required = param.default == param.empty
            
            # Get default value
            default_value = None if param.default == param.empty else param.default
            
            # Get description from docstring
            description = param_descriptions.get(param_name, f"Parameter {param_name}")
            
            parameters.append(ToolParameter(
                name=param_name,
                type=param_type,
                description=description,
                required=required,
                default=default_value
            ))
        
        return parameters
    
    def _get_parameter_type(self, param: inspect.Parameter, type_hint: Any) -> ParameterType:
        """Determine parameter type from signature and type hints"""
        # Check type hint first
        if type_hint:
            if type_hint == str or type_hint == 'str':
                return ParameterType.STRING
            elif type_hint == int or type_hint == 'int':
                return ParameterType.INTEGER
            elif type_hint == float or type_hint == 'float':
                return ParameterType.FLOAT
            elif type_hint == bool or type_hint == 'bool':
                return ParameterType.BOOLEAN
            elif hasattr(type_hint, '__origin__'):
                # Handle generic types like List, Dict, etc.
                origin = type_hint.__origin__
                if origin == list:
                    return ParameterType.ARRAY
                elif origin == dict:
                    return ParameterType.OBJECT
        
        # Check default value type
        if param.default != param.empty:
            if isinstance(param.default, str):
                return ParameterType.STRING
            elif isinstance(param.default, int):
                return ParameterType.INTEGER
            elif isinstance(param.default, float):
                return ParameterType.NUMBER
            elif isinstance(param.default, bool):
                return ParameterType.BOOLEAN
            elif isinstance(param.default, list):
                return ParameterType.ARRAY
            elif isinstance(param.default, dict):
                return ParameterType.OBJECT
        
        # Default to string
        return ParameterType.STRING
    
    def _parse_docstring_parameters(self, func: Callable) -> Dict[str, str]:
        """Parse parameter descriptions from docstring"""
        docstring = inspect.getdoc(func)
        if not docstring:
            return {}
        
        param_descriptions = {}
        lines = docstring.split('\n')
        
        # Look for Args/Arguments/Parameters section
        in_args_section = False
        for line in lines:
            line = line.strip()
            
            # Check if we're entering args section
            if line.lower().startswith(('args:', 'arguments:', 'parameters:')):
                in_args_section = True
                continue
            
            # Check if we're leaving args section
            if in_args_section and line.lower().startswith(('returns:', 'return:', 'raises:', 'examples:', 'note:')):
                break
            
            # Parse parameter line
            if in_args_section and line:
                # Look for pattern: param_name (type): description
                # or: param_name: description
                if ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        param_part = parts[0].strip()
                        description = parts[1].strip()
                        
                        # Extract parameter name (remove type annotation if present)
                        if '(' in param_part and ')' in param_part:
                            param_name = param_part.split('(')[0].strip()
                        else:
                            param_name = param_part
                        
                        # Clean parameter name (remove leading dashes, etc.)
                        param_name = param_name.lstrip('- \t')
                        
                        if param_name:
                            param_descriptions[param_name] = description
        
        return param_descriptions
    
    async def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        """Execute the wrapped function"""
        try:
            # Validate and prepare arguments
            validated_args = self._prepare_arguments(arguments)
            
            # Execute function
            if self.is_async:
                result = await self.func(**validated_args)
            else:
                # Run sync function in executor to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: self.func(**validated_args))
            
            return ToolResult(
                success=True,
                data=result,
                metadata={
                    "function_name": self.func.__name__,
                    "arguments_used": validated_args
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Function execution error: {str(e)}",
                metadata={
                    "function_name": self.func.__name__,
                    "arguments_attempted": arguments
                }
            )
    
    def _prepare_arguments(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare and validate arguments for function call"""
        validated_args = {}
        
        # Get function signature
        sig = inspect.signature(self.func)
        
        for param_name, param in sig.parameters.items():
            # Skip *args and **kwargs
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            
            if param_name in arguments:
                # Use provided argument
                validated_args[param_name] = arguments[param_name]
            elif param.default != param.empty:
                # Use default value
                validated_args[param_name] = param.default
            else:
                # Required parameter missing
                raise ValueError(f"Required parameter '{param_name}' is missing")
        
        return validated_args


class FunctionRegistry:
    """Registry for managing function-based tools"""
    
    def __init__(self):
        self.functions: Dict[str, Callable] = {}
        self.tools: Dict[str, FunctionToolAdapter] = {}
    
    def register_function(
        self, 
        func: Callable, 
        name: Optional[str] = None, 
        description: Optional[str] = None
    ) -> FunctionToolAdapter:
        """
        Register a function as a tool
        
        Args:
            func: Function to register
            name: Optional custom name
            description: Optional custom description
            
        Returns:
            FunctionToolAdapter instance
        """
        tool_name = name or func.__name__
        
        # Create tool adapter
        tool = FunctionToolAdapter(func, name, description)
        
        # Store in registry
        self.functions[tool_name] = func
        self.tools[tool_name] = tool
        
        return tool
    
    def register_functions(self, functions: List[Callable]) -> List[FunctionToolAdapter]:
        """Register multiple functions"""
        tools = []
        for func in functions:
            tool = self.register_function(func)
            tools.append(tool)
        return tools
    
    def get_tool(self, name: str) -> Optional[FunctionToolAdapter]:
        """Get tool by name"""
        return self.tools.get(name)
    
    def get_all_tools(self) -> List[FunctionToolAdapter]:
        """Get all registered tools"""
        return list(self.tools.values())
    
    def get_function(self, name: str) -> Optional[Callable]:
        """Get original function by name"""
        return self.functions.get(name)
    
    def unregister(self, name: str) -> bool:
        """Unregister a function/tool"""
        if name in self.tools:
            del self.tools[name]
            del self.functions[name]
            return True
        return False
    
    def clear(self):
        """Clear all registered functions/tools"""
        self.functions.clear()
        self.tools.clear()
    
    def list_functions(self) -> List[str]:
        """List all registered function names"""
        return list(self.functions.keys())


def create_tool_from_function(
    func: Callable, 
    name: Optional[str] = None, 
    description: Optional[str] = None
) -> FunctionToolAdapter:
    """
    Convenience function to create a tool from a function
    
    Args:
        func: Function to convert
        name: Optional custom name
        description: Optional custom description
        
    Returns:
        FunctionToolAdapter instance
    """
    return FunctionToolAdapter(func, name, description)

