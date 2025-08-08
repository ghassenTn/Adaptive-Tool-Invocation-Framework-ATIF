"""
Simple Argument Generator for the Enhanced Tool Calling Framework
(No external dependencies version)
"""

import re
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime

from .base_tool import BaseTool, ParameterType


@dataclass
class ArgumentExtractionResult:
    """Result of argument extraction process"""
    arguments: Dict[str, Any]
    confidence_score: float
    missing_required: List[str]
    extraction_details: Dict[str, str]


class SimpleArgumentGenerator:
    """
    Simple argument generator that extracts and validates arguments from user queries
    using basic pattern matching and keyword extraction.
    
    This version doesn't require external dependencies like spaCy.
    """
    
    def __init__(self):
        """Initialize the simple argument generator."""
        self.extraction_patterns = self._setup_patterns()
    
    def _setup_patterns(self) -> Dict[str, List[str]]:
        """Setup extraction patterns for common argument types"""
        return {
            "numbers": [
                r'\b(\d+\.?\d*)\b',  # Integer or float
                r'\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\b'  # Numbers with commas
            ],
            "operations": [
                r'\b(add|addition|plus|\+)\b',
                r'\b(subtract|subtraction|minus|\-)\b',
                r'\b(multiply|multiplication|times|\*|x)\b',
                r'\b(divide|division|divided by|\/)\b',
                r'\b(power|to the power of|\^|\*\*)\b',
                r'\b(sqrt|square root)\b'
            ],
            "locations": [
                r'\bفي\s+([\u0600-\u06FF\s]+)\b', # "في صفاقس"
                r'\b(مدينة|مدينه)\s+([\u0600-\u06FF\s]+)\b', # "مدينة صفاقس"
                r'\b(في|ب|لـ)\s*([\u0600-\u06FF\s]+)\b', # "في صفاقس", "بصفاقس", "لصفاقس"
                r'\b(لـ|في|ب|عن)\s*([\u0600-\u06FF\s]+)\b', # "عن لندن"
                r'\bin\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',  # "in New York"
                r'\bfor\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',  # "for London"
                r'\bat\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',   # "at Paris"
                r'\b(مدينة|city)\s+([A-Za-z\s]+)\b', # "مدينة لندن", "city London"
                r'\b(الطقس في|weather in)\s+([A-Za-z\u0600-\u06FF\s]+)\b' # "الطقس في لندن", "weather in London"
            ],
            "units": [
                r'\b(celsius|fahrenheit|c|f)\b',
                r'\b(metric|imperial)\b'
            ],
            "text_content": [
                r'text[:\s]+["\\]([^"]\\]+)["\\]',  # "text: 'content'"
                r'analyze[:\s]+["\\]([^"]\\]+)["\\]',  # "analyze: 'content'"
                r'["\\]([^"]\\]{10,})["\\]'  # Any quoted text longer than 10 chars
            ],
            "filenames": [
                r'\b(\w+\.\w+)\b',  # filename.extension
                r'\bfile\s+([^\s]+)\b',  # "file filename"
                r'\babout\s+([^\s]+\.\w+)\b'  # "about document.pdf"
            ],
            "time_operations": [
                r'\b(current|now|today|time|date|timestamp)\b'
            ],
            "analysis_types": [
                r'\b(basic|detailed|sentiment)\b'
            ],
            "boolean_values": [
                r'\b(true|yes|1|on|enable|enabled)\b',
                r'\b(false|no|0|off|disable|disabled)\b'
            ]
        }
    
    def generate_arguments(
        self, 
        user_query: str, 
        tool: BaseTool, 
        context: Optional[Dict[str, Any]] = None
    ) -> ArgumentExtractionResult:
        """
        Generate arguments for a tool based on user query.
        
        Args:
            user_query: User's natural language query
            tool: Tool that needs arguments
            context: Optional context information
            
        Returns:
            ArgumentExtractionResult with extracted arguments
        """
        arguments = {}
        extraction_details = {}
        missing_required = []
        confidence_scores = []
        
        for param in tool.parameters:
            extracted_value, confidence, details = self._extract_parameter(
                param, user_query, context
            )
            
            if extracted_value is not None:
                try:
                    # Validate the extracted value
                    validated_value = tool._validate_parameter_value(param, extracted_value)
                    arguments[param.name] = validated_value
                    extraction_details[param.name] = details
                    confidence_scores.append(confidence)
                except ValueError as e:
                    extraction_details[param.name] = f"Validation failed: {str(e)}"
                    if param.required:
                        missing_required.append(param.name)
            else:
                if param.required and param.default is None:
                    missing_required.append(param.name)
                elif param.default is not None:
                    arguments[param.name] = param.default
                    extraction_details[param.name] = "Used default value"
                    confidence_scores.append(1.0)
        
        # Calculate overall confidence
        overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        # Reduce confidence if required parameters are missing
        if missing_required:
            overall_confidence *= 0.5
        
        return ArgumentExtractionResult(
            arguments=arguments,
            confidence_score=overall_confidence,
            missing_required=missing_required,
            extraction_details=extraction_details
        )
    
    def _extract_parameter(
        self, 
        param, 
        user_query: str, 
        context: Optional[Dict[str, Any]]
    ) -> Tuple[Any, float, str]:
        """
        Extract a specific parameter from the user query.
        
        Returns:
            Tuple of (extracted_value, confidence_score, extraction_details)
        """
        param_name_lower = param.name.lower()
        query_lower = user_query.lower()
        
        # Strategy 1: Direct keyword matching
        if param_name_lower in query_lower:
            value, confidence, details = self._extract_by_keyword(param, user_query)
            if value is not None:
                return value, confidence, details
        
        # Strategy 2: Pattern-based extraction
        value, confidence, details = self._extract_by_pattern(param, user_query)
        if value is not None:
                return value, confidence, details
        
        # Strategy 3: Context-based extraction
        if context:
            value, confidence, details = self._extract_from_context(param, context)
            if value is not None:
                return value, confidence, details
        
        # Strategy 4: Tool-specific extraction
        value, confidence, details = self._extract_tool_specific(param, user_query)
        if value is not None:
            return value, confidence, details
        
        return None, 0.0, "No extraction method succeeded"
    
    def _extract_by_keyword(self, param, user_query: str) -> Tuple[Any, float, str]:
        """Extract parameter value by looking for keywords around parameter name"""
        param_name_lower = param.name.lower()
        query_lower = user_query.lower()
        
        # Find parameter name in query
        param_index = query_lower.find(param_name_lower)
        if param_index == -1:
            return None, 0.0, "Parameter name not found"
        
        # Look for value after parameter name
        after_param = user_query[param_index + len(param_name_lower):].strip()
        
        # Remove common separators
        for sep in [":", "=", "is", "to", "of", "by"]:
            if after_param.lower().startswith(sep):
                after_param = after_param[len(sep):].strip()
                break
        
        # Extract value based on parameter type
        if param.type == ParameterType.STRING:
            # Look for quoted strings first
            quoted_match = re.match(r'^[\"\\]([^\"\\]*)[\"\\]', after_param)
            if quoted_match:
                return quoted_match.group(1), 0.9, "Extracted quoted string"
            
            # Take the next word
            word_match = re.match(r'^(\S+)', after_param)
            if word_match:
                return word_match.group(1), 0.7, "Extracted first word"
        
        elif param.type in [ParameterType.INTEGER, ParameterType.FLOAT]:
            number_match = re.match(r'^(\d+\.?\d*)', after_param)
            if number_match:
                value = float(number_match.group(1)) if '.' in number_match.group(1) else int(number_match.group(1))
                return value, 0.9, "Extracted number after parameter name"
        
        elif param.type == ParameterType.BOOLEAN:
            for true_val in ["true", "yes", "1", "on", "enable"]:
                if after_param.lower().startswith(true_val):
                    return True, 0.8, f"Extracted boolean (true) from '{true_val}'"
            
            for false_val in ["false", "no", "0", "off", "disable"]:
                if after_param.lower().startswith(false_val):
                    return False, 0.8, f"Extracted boolean (false) from '{false_val}'"
        
        return None, 0.0, "Could not extract value after parameter name"
    
    def _extract_by_pattern(self, param, user_query: str) -> Tuple[Any, float, str]:
        """Extract parameter value using predefined patterns"""
        query_lower = user_query.lower()
        
        if param.type in [ParameterType.INTEGER, ParameterType.FLOAT]:
            # Special handling for rectangle area calculation
            if param.name.lower() == "length":
                # Look for patterns like "طوله 10" or "length 10"
                length_patterns = [
                    r'(?:طوله|الطول|length)\s*(\d+\.?\d*)',
                    r'مستطيل\s+طوله\s*(\d+\.?\d*)',
                    r'rectangle.*length\s*(\d+\.?\d*)'
                ]
                for pattern in length_patterns:
                    match = re.search(pattern, user_query, re.IGNORECASE)
                    if match:
                        value = float(match.group(1)) if '.' in match.group(1) else int(match.group(1))
                        return value, 0.8, f"Extracted length: {match.group(1)}"
            
            elif param.name.lower() == "width":
                # Look for patterns like "وعرضه 5" or "width 5"
                width_patterns = [
                    r'(?:وعرضه|العرض|width)\s*(\d+\.?\d*)',
                    r'مستطيل.*وعرضه\s*(\d+\.?\d*)',
                    r'rectangle.*width\s*(\d+\.?\d*)'
                ]
                for pattern in width_patterns:
                    match = re.search(pattern, user_query, re.IGNORECASE)
                    if match:
                        value = float(match.group(1)) if '.' in match.group(1) else int(match.group(1))
                        return value, 0.8, f"Extracted width: {match.group(1)}"
            
            # General number extraction
            for pattern in self.extraction_patterns["numbers"]:
                matches = re.findall(pattern, user_query)
                if matches:
                    try:
                        # Clean the number (remove commas)
                        clean_number = matches[0].replace(',', '')
                        value = float(clean_number) if '.' in clean_number else int(clean_number)
                        return value, 0.6, f"Extracted number from pattern: {matches[0]}"
                    except ValueError:
                        continue
        
        elif param.type == ParameterType.STRING:
            # Check parameter-specific patterns
            if param.name.lower() == "cityname":
                for pattern in self.extraction_patterns["locations"]:
                    match = re.search(pattern, user_query, re.IGNORECASE)
                    if match:
                        return match.group(1), 0.7, f"Extracted city name: {match.group(1)}"

            if "location" in param.description.lower() or "place" in param.description.lower():
                for pattern in self.extraction_patterns["locations"]:
                    match = re.search(pattern, user_query, re.IGNORECASE)
                    if match:
                        return match.group(1), 0.7, f"Extracted location: {match.group(1)}"
            
            if "file" in param.description.lower() or "filename" in param.description.lower():
                for pattern in self.extraction_patterns["filenames"]:
                    match = re.search(pattern, user_query)
                    if match:
                        return match.group(1), 0.8, f"Extracted filename: {match.group(1)}"
            
            if "text" in param.description.lower() or "content" in param.description.lower():
                for pattern in self.extraction_patterns["text_content"]:
                    match = re.search(pattern, user_query, re.IGNORECASE)
                    if match:
                        return match.group(1), 0.8, f"Extracted text content: {match.group(1)[:30]}..."
            
            # Check for enum values
            if param.enum_values:
                for enum_val in param.enum_values:
                    if str(enum_val).lower() in query_lower:
                        return enum_val, 0.9, f"Found enum value: {enum_val}"
        
        elif param.type == ParameterType.BOOLEAN:
            for pattern in self.extraction_patterns["boolean_values"]:
                match = re.search(pattern, query_lower)
                if match:
                    true_values = ["true", "yes", "1", "on", "enable", "enabled"]
                    is_true = match.group(1) in true_values
                    return is_true, 0.7, f"Extracted boolean from: {match.group(1)}"
        
        return None, 0.0, "No pattern matches found"
    
    def _extract_from_context(self, param, context: Dict[str, Any]) -> Tuple[Any, float, str]:
        """Extract parameter value from context"""
        if param.name in context:
            return context[param.name], 1.0, "Extracted from context"
        
        # Look for similar keys in context
        param_name_lower = param.name.lower()
        for key, value in context.items():
            if param_name_lower in key.lower() or key.lower() in param_name_lower:
                return value, 0.7, f"Extracted from context key '{key}'"
        
        return None, 0.0, "Not found in context"
    
    def _extract_tool_specific(self, param, user_query: str) -> Tuple[Any, float, str]:
        """Extract parameter using tool-specific logic"""
        query_lower = user_query.lower()
        
        # Calculator-specific extraction
        if param.name == "operation":
            for pattern in self.extraction_patterns["operations"]:
                match = re.search(pattern, query_lower)
                if match:
                    operation_map = {
                        "add": "add", "addition": "add", "plus": "add", "+": "add",
                        "subtract": "subtract", "subtraction": "subtract", "minus": "subtract", "-": "subtract",
                        "multiply": "multiply", "multiplication": "multiply", "times": "multiply", "*": "multiply", "x": "multiply",
                        "divide": "divide", "division": "divide", "divided by": "divide", "/": "divide",
                        "power": "power", "to the power of": "power", "^": "power", "**": "power",
                        "sqrt": "sqrt", "square root": "sqrt"
                    }
                    
                    matched_text = match.group(1)
                    operation = operation_map.get(matched_text, matched_text)
                    return operation, 0.8, f"Extracted operation: {operation}"
        
        # Weather-specific extraction
        elif param.name == "units":
            for pattern in self.extraction_patterns["units"]:
                match = re.search(pattern, query_lower)
                if match:
                    unit_map = {
                        "celsius": "celsius", "c": "celsius",
                        "fahrenheit": "fahrenheit", "f": "fahrenheit"
                    }
                    unit = unit_map.get(match.group(1), match.group(1))
                    return unit, 0.8, f"Extracted unit: {unit}"
        
        # Time tool specific extraction
        elif param.name == "operation" and "time" in query_lower:
            time_ops = ["current_time", "current_date", "timestamp", "format_time"]
            for op in time_ops:
                op_keywords = op.split("_")
                if all(keyword in query_lower for keyword in op_keywords):
                    return op, 0.8, f"Extracted time operation: {op}"
            
            # Fallback based on keywords
            if any(word in query_lower for word in ["time", "clock"]):
                return "current_time", 0.6, "Inferred current_time operation"
            elif any(word in query_lower for word in ["date", "today"]):
                return "current_date", 0.6, "Inferred current_date operation"
        
        # Text analyzer specific extraction
        elif param.name == "analysis_type":
            for pattern in self.extraction_patterns["analysis_types"]:
                match = re.search(pattern, query_lower)
                if match:
                    return match.group(1), 0.8, f"Extracted analysis type: {match.group(1)}"
            
            # Infer from context
            if "sentiment" in query_lower:
                return "sentiment", 0.7, "Inferred sentiment analysis"
            elif "detailed" in query_lower or "detail" in query_lower:
                return "detailed", 0.7, "Inferred detailed analysis"
        
        return None, 0.0, "No tool-specific extraction possible"
    
    def suggest_missing_arguments(
        self, 
        tool: BaseTool, 
        missing_params: List[str]
    ) -> List[str]:
        """
        Suggest how to provide missing arguments.
        
        Args:
            tool: Tool that needs arguments
            missing_params: List of missing parameter names
            
        Returns:
            List of suggestion strings
        """
        suggestions = []
        
        for param_name in missing_params:
            param = next((p for p in tool.parameters if p.name == param_name), None)
            if param:
                if param.type == ParameterType.STRING:
                    suggestions.append(f"Please provide a value for '{param_name}': {param.description}")
                elif param.type in [ParameterType.INTEGER, ParameterType.FLOAT]:
                    suggestions.append(f"Please provide a number for '{param_name}': {param.description}")
                elif param.type == ParameterType.BOOLEAN:
                    suggestions.append(f"Please specify true/false for '{param_name}': {param.description}")
                elif param.type == ParameterType.ARRAY:
                    suggestions.append(f"Please provide a list for '{param_name}': {param.description}")
                else:
                    suggestions.append(f"Please provide '{param_name}': {param.description}")
        
        return suggestions

