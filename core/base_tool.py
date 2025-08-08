"""
Base Tool class for the Enhanced Tool Calling Framework
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum


class ParameterType(Enum):
    """Supported parameter types for tools"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


@dataclass
class ToolParameter:
    """Represents a tool parameter with its metadata"""
    name: str
    type: ParameterType
    description: str
    required: bool = True
    default: Optional[Any] = None
    enum_values: Optional[List[Any]] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None


@dataclass
class ToolResult:
    """Represents the result of a tool execution"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseTool(ABC):
    """
    Abstract base class for all tools in the Enhanced Tool Calling Framework.
    
    This class defines the interface that all tools must implement to be
    compatible with the framework.
    """
    
    def __init__(self, name: str, description: str, parameters: List[ToolParameter]):
        """
        Initialize the base tool.
        
        Args:
            name: Unique identifier for the tool
            description: Human-readable description of what the tool does
            parameters: List of parameters the tool accepts
        """
        self.name = name
        self.description = description
        self.parameters = parameters
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Validate that parameters are properly defined"""
        param_names = [p.name for p in self.parameters]
        if len(param_names) != len(set(param_names)):
            raise ValueError(f"Duplicate parameter names found in tool '{self.name}'")
    
    @abstractmethod
    async def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        """
        Execute the tool with the given arguments.
        
        Args:
            arguments: Dictionary of parameter names to values
            
        Returns:
            ToolResult containing the execution result
        """
        pass
    
    def validate_arguments(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize the provided arguments.
        
        Args:
            arguments: Raw arguments to validate
            
        Returns:
            Validated and normalized arguments
            
        Raises:
            ValueError: If validation fails
        """
        validated = {}
        
        # Check required parameters
        required_params = {p.name for p in self.parameters if p.required}
        provided_params = set(arguments.keys())
        missing_params = required_params - provided_params
        
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")
        
        # Validate each parameter
        for param in self.parameters:
            if param.name in arguments:
                value = arguments[param.name]
                validated[param.name] = self._validate_parameter_value(param, value)
            elif param.default is not None:
                validated[param.name] = param.default
        
        return validated
    
    def _validate_parameter_value(self, param: ToolParameter, value: Any) -> Any:
        """Validate a single parameter value"""
        # Type validation
        if param.type == ParameterType.STRING and not isinstance(value, str):
            raise ValueError(f"Parameter '{param.name}' must be a string")
        elif param.type == ParameterType.INTEGER and not isinstance(value, int):
            raise ValueError(f"Parameter '{param.name}' must be an integer")
        elif param.type == ParameterType.FLOAT and not isinstance(value, (int, float)):
            raise ValueError(f"Parameter '{param.name}' must be a number")
        elif param.type == ParameterType.BOOLEAN and not isinstance(value, bool):
            raise ValueError(f"Parameter '{param.name}' must be a boolean")
        elif param.type == ParameterType.ARRAY and not isinstance(value, list):
            raise ValueError(f"Parameter '{param.name}' must be an array")
        elif param.type == ParameterType.OBJECT and not isinstance(value, dict):
            raise ValueError(f"Parameter '{param.name}' must be an object")
        
        # Enum validation
        if param.enum_values and value not in param.enum_values:
            raise ValueError(f"Parameter '{param.name}' must be one of {param.enum_values}")
        
        # Range validation for numeric types
        if param.type in [ParameterType.INTEGER, ParameterType.FLOAT]:
            if param.min_value is not None and value < param.min_value:
                raise ValueError(f"Parameter '{param.name}' must be >= {param.min_value}")
            if param.max_value is not None and value > param.max_value:
                raise ValueError(f"Parameter '{param.name}' must be <= {param.max_value}")
        
        return value
    
    def get_schema(self) -> Dict[str, Any]:
        """
        Get the JSON schema representation of this tool.
        
        Returns:
            Dictionary representing the tool's schema
        """
        properties = {}
        required = []
        
        for param in self.parameters:
            prop = {
                "type": param.type.value,
                "description": param.description
            }
            
            if param.enum_values:
                prop["enum"] = param.enum_values
            
            if param.type in [ParameterType.INTEGER, ParameterType.FLOAT]:
                if param.min_value is not None:
                    prop["minimum"] = param.min_value
                if param.max_value is not None:
                    prop["maximum"] = param.max_value
            
            if param.default is not None:
                prop["default"] = param.default
            
            properties[param.name] = prop
            
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    
    def __str__(self) -> str:
        return f"Tool(name='{self.name}', description='{self.description}')"
    
    def __repr__(self) -> str:
        return self.__str__()

