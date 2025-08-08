"""
Tool Argument Generator for the Enhanced Tool Calling Framework
"""

import re
import json
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import spacy
from spacy.matcher import Matcher

from .base_tool import BaseTool, ParameterType


@dataclass
class ArgumentExtractionResult:
    """Result of argument extraction process"""
    arguments: Dict[str, Any]
    confidence_score: float
    missing_required: List[str]
    extraction_details: Dict[str, str]


class ToolArgumentGenerator:
    """
    Tool argument generator that extracts and validates arguments from user queries
    for less powerful LLMs that struggle with precise argument generation.
    """
    
    def __init__(self, nlp_model: str = "en_core_web_sm"):
        """
        Initialize the argument generator.
        
        Args:
            nlp_model: spaCy model name for NLP processing
        """
        try:
            self.nlp = spacy.load(nlp_model)
        except OSError:
            # Fallback to a basic model if the specified one isn't available
            self.nlp = spacy.blank("en")
        
        self.matcher = Matcher(self.nlp.vocab)
        self._setup_patterns()
    
    def _setup_patterns(self):
        """Setup spaCy patterns for common argument types"""
        # Number patterns
        number_pattern = [{"LIKE_NUM": True}]
        self.matcher.add("NUMBER", [number_pattern])
        
        # Date patterns
        date_patterns = [
            [{"TEXT": {"REGEX": r"\d{1,2}/\d{1,2}/\d{4}"}}],  # MM/DD/YYYY
            [{"TEXT": {"REGEX": r"\d{4}-\d{1,2}-\d{1,2}"}}],  # YYYY-MM-DD
            [{"TEXT": "today"}],
            [{"TEXT": "tomorrow"}],
            [{"TEXT": "yesterday"}]
        ]
        for pattern in date_patterns:
            self.matcher.add("DATE", [pattern])
        
        # Email patterns
        email_pattern = [{"TEXT": {"REGEX": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"}}]
        self.matcher.add("EMAIL", [email_pattern])
        
        # URL patterns
        url_pattern = [{"TEXT": {"REGEX": r"https?://[^\s]+"}}]
        self.matcher.add("URL", [url_pattern])
    
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
        doc = self.nlp(user_query)
        matches = self.matcher(doc)
        
        arguments = {}
        extraction_details = {}
        missing_required = []
        confidence_scores = []
        
        for param in tool.parameters:
            extracted_value, confidence, details = self._extract_parameter(
                param, user_query, doc, matches, context
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
        doc, 
        matches, 
        context: Optional[Dict[str, Any]]
    ) -> tuple[Any, float, str]:
        """
        Extract a specific parameter from the user query.
        
        Returns:
            Tuple of (extracted_value, confidence_score, extraction_details)
        """
        param_name_lower = param.name.lower()
        query_lower = user_query.lower()
        
        # Strategy 1: Direct keyword matching
        if param_name_lower in query_lower:
            value, confidence, details = self._extract_by_keyword(param, user_query, doc)
            if value is not None:
                return value, confidence, details
        
        # Strategy 2: Pattern-based extraction
        value, confidence, details = self._extract_by_pattern(param, user_query, doc, matches)
        if value is not None:
            return value, confidence, details
        
        # Strategy 3: Context-based extraction
        if context:
            value, confidence, details = self._extract_from_context(param, context)
            if value is not None:
                return value, confidence, details
        
        # Strategy 4: Semantic extraction
        value, confidence, details = self._extract_by_semantics(param, user_query, doc)
        if value is not None:
            return value, confidence, details
        
        return None, 0.0, "No extraction method succeeded"
    
    def _extract_by_keyword(self, param, user_query: str, doc) -> tuple[Any, float, str]:
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
        for sep in [":", "=", "is", "to", "of"]:
            if after_param.lower().startswith(sep):
                after_param = after_param[len(sep):].strip()
                break
        
        # Extract value based on parameter type
        if param.type == ParameterType.STRING:
            # Take the next word or quoted string
            match = re.match(r'^"([^"]*)"', after_param)
            if match:
                return match.group(1), 0.9, "Extracted quoted string"
            
            match = re.match(r'^(\S+)', after_param)
            if match:
                return match.group(1), 0.7, "Extracted first word"
        
        elif param.type in [ParameterType.INTEGER, ParameterType.FLOAT]:
            match = re.match(r'^(\d+\.?\d*)', after_param)
            if match:
                value = float(match.group(1)) if '.' in match.group(1) else int(match.group(1))
                return value, 0.9, "Extracted number after parameter name"
        
        elif param.type == ParameterType.BOOLEAN:
            for true_val in ["true", "yes", "1", "on", "enable"]:
                if after_param.lower().startswith(true_val):
                    return True, 0.8, f"Extracted boolean (true) from '{true_val}'"
            
            for false_val in ["false", "no", "0", "off", "disable"]:
                if after_param.lower().startswith(false_val):
                    return False, 0.8, f"Extracted boolean (false) from '{false_val}'"
        
        return None, 0.0, "Could not extract value after parameter name"
    
    def _extract_by_pattern(self, param, user_query: str, doc, matches) -> tuple[Any, float, str]:
        """Extract parameter value using predefined patterns"""
        if param.type in [ParameterType.INTEGER, ParameterType.FLOAT]:
            # Look for numbers in the query
            numbers = []
            for match_id, start, end in matches:
                if self.nlp.vocab.strings[match_id] == "NUMBER":
                    token = doc[start:end]
                    try:
                        value = float(token.text) if '.' in token.text else int(token.text)
                        numbers.append(value)
                    except ValueError:
                        continue
            
            if numbers:
                # Return the first number found
                return numbers[0], 0.6, f"Extracted number from pattern matching"
        
        elif param.type == ParameterType.STRING:
            # Look for emails, URLs, etc.
            for match_id, start, end in matches:
                match_label = self.nlp.vocab.strings[match_id]
                if match_label in ["EMAIL", "URL"]:
                    value = doc[start:end].text
                    return value, 0.8, f"Extracted {match_label.lower()} from pattern"
        
        return None, 0.0, "No pattern matches found"
    
    def _extract_from_context(self, param, context: Dict[str, Any]) -> tuple[Any, float, str]:
        """Extract parameter value from context"""
        if param.name in context:
            return context[param.name], 1.0, "Extracted from context"
        
        # Look for similar keys in context
        param_name_lower = param.name.lower()
        for key, value in context.items():
            if param_name_lower in key.lower() or key.lower() in param_name_lower:
                return value, 0.7, f"Extracted from context key '{key}'"
        
        return None, 0.0, "Not found in context"
    
    def _extract_by_semantics(self, param, user_query: str, doc) -> tuple[Any, float, str]:
        """Extract parameter value using semantic understanding"""
        param_desc_lower = param.description.lower()
        
        # Common semantic patterns
        if "location" in param_desc_lower or "place" in param_desc_lower:
            # Look for location entities
            for ent in doc.ents:
                if ent.label_ in ["GPE", "LOC"]:  # Geopolitical entity or location
                    return ent.text, 0.7, f"Extracted location entity: {ent.text}"
        
        elif "person" in param_desc_lower or "name" in param_desc_lower:
            # Look for person entities
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    return ent.text, 0.7, f"Extracted person entity: {ent.text}"
        
        elif "date" in param_desc_lower or "time" in param_desc_lower:
            # Look for date/time entities
            for ent in doc.ents:
                if ent.label_ in ["DATE", "TIME"]:
                    return ent.text, 0.7, f"Extracted date/time entity: {ent.text}"
        
        elif "money" in param_desc_lower or "price" in param_desc_lower:
            # Look for money entities
            for ent in doc.ents:
                if ent.label_ == "MONEY":
                    return ent.text, 0.7, f"Extracted money entity: {ent.text}"
        
        return None, 0.0, "No semantic extraction possible"
    
    def suggest_missing_arguments(
        self, 
        tool: BaseTool, 
        missing_params: List[str]
    ) -> Dict[str, str]:
        """
        Generate suggestions for missing required arguments.
        
        Args:
            tool: Tool that needs arguments
            missing_params: List of missing parameter names
            
        Returns:
            Dictionary mapping parameter names to suggestion strings
        """
        suggestions = {}
        
        for param_name in missing_params:
            param = next((p for p in tool.parameters if p.name == param_name), None)
            if param:
                suggestion = f"Please provide a value for '{param_name}': {param.description}"
                
                if param.enum_values:
                    suggestion += f" (options: {', '.join(map(str, param.enum_values))})"
                
                if param.type == ParameterType.STRING:
                    suggestion += " (text value)"
                elif param.type in [ParameterType.INTEGER, ParameterType.FLOAT]:
                    suggestion += " (numeric value)"
                    if param.min_value is not None or param.max_value is not None:
                        range_info = []
                        if param.min_value is not None:
                            range_info.append(f"min: {param.min_value}")
                        if param.max_value is not None:
                            range_info.append(f"max: {param.max_value}")
                        suggestion += f" ({', '.join(range_info)})"
                elif param.type == ParameterType.BOOLEAN:
                    suggestion += " (true/false)"
                
                suggestions[param_name] = suggestion
        
        return suggestions

