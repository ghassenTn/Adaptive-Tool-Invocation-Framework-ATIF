"""
Google Gemini Wrapper for Enhanced Tool Calling Framework
"""

import os
import json
import asyncio
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from .llm_wrapper import BaseLLMWrapper, LLMResponse


class GeminiWrapper(BaseLLMWrapper):
    """Wrapper for Google Gemini models"""
    
    def __init__(
        self, 
        model_name: str = "gemini-1.5-flash", 
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Gemini wrapper
        
        Args:
            model_name: Gemini model name (e.g., "gemini-1.5-flash", "gemini-1.5-pro")
            api_key: Google AI API key (if None, will try to read from environment)
            **kwargs: Additional configuration parameters
        """
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "google-generativeai is required for Gemini integration. "
                "Install it with: pip install google-generativeai"
            )
        
        super().__init__(model_name, **kwargs)
        
        # Get API key from parameter or environment
        self.api_key = api_key or self._get_api_key()
        if not self.api_key:
            raise ValueError(
                "Google AI API key is required. Set GOOGLE_AI_API_KEY environment variable "
                "or pass api_key parameter"
            )
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Initialize model
        generation_config = {
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.95),
            "top_k": kwargs.get("top_k", 40),
            "max_output_tokens": kwargs.get("max_tokens", 1000),
        }
        
        safety_settings = kwargs.get("safety_settings", [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ])
        
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        self.generation_config = generation_config
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key from environment variables"""
        # Try different environment variable names
        env_vars = [
            "GOOGLE_AI_API_KEY",
            "GEMINI_API_KEY", 
            "GOOGLE_API_KEY",
            "GENAI_API_KEY"
        ]
        
        for var in env_vars:
            api_key = os.getenv(var)
            if api_key:
                return api_key
        
        return None
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using Gemini API"""
        try:
            # Update generation config with any provided kwargs
            config = self.generation_config.copy()
            if "temperature" in kwargs:
                config["temperature"] = kwargs["temperature"]
            if "max_tokens" in kwargs:
                config["max_output_tokens"] = kwargs["max_tokens"]
            if "top_p" in kwargs:
                config["top_p"] = kwargs["top_p"]
            if "top_k" in kwargs:
                config["top_k"] = kwargs["top_k"]
            
            # Create a new model instance with updated config if needed
            if config != self.generation_config:
                model = genai.GenerativeModel(
                    model_name=self.model_name,
                    generation_config=config,
                    safety_settings=self.model._safety_settings
                )
            else:
                model = self.model
            
            # Generate response
            response = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: model.generate_content(prompt)
            )
            
            # Extract content
            content = ""
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    content = "".join(part.text for part in candidate.content.parts if part.text)
            
            # Extract usage information if available
            usage = None
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = {
                    "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0),
                    "completion_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0),
                    "total_tokens": getattr(response.usage_metadata, 'total_token_count', 0)
                }
            
            # Extract finish reason
            finish_reason = "stop"
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason'):
                    finish_reason = str(candidate.finish_reason).lower()
            
            return LLMResponse(
                content=content,
                usage=usage,
                model=self.model_name,
                finish_reason=finish_reason
            )
            
        except Exception as e:
            raise Exception(f"Gemini API error: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Gemini model information"""
        return {
            "provider": "Google",
            "model": self.model_name,
            "type": "cloud",
            "supports_streaming": True,
            "max_context_length": self._get_context_length(),
            "supports_multimodal": self._supports_multimodal()
        }
    
    def _get_context_length(self) -> int:
        """Get context length for different Gemini models"""
        context_lengths = {
            "gemini-1.5-flash": 1000000,  # 1M tokens
            "gemini-1.5-pro": 2000000,    # 2M tokens
            "gemini-1.0-pro": 32768,      # 32K tokens
            "gemini-pro": 32768,          # 32K tokens
            "gemini-pro-vision": 16384    # 16K tokens
        }
        return context_lengths.get(self.model_name, 32768)
    
    def _supports_multimodal(self) -> bool:
        """Check if model supports multimodal input"""
        multimodal_models = [
            "gemini-1.5-flash",
            "gemini-1.5-pro", 
            "gemini-pro-vision"
        ]
        return self.model_name in multimodal_models
    
    async def generate_with_tools(
        self, 
        prompt: str, 
        tools: List[Dict[str, Any]], 
        **kwargs
    ) -> LLMResponse:
        """
        Generate response with function calling capabilities
        
        Args:
            prompt: Input prompt
            tools: List of tool definitions in Gemini format
            **kwargs: Additional generation parameters
            
        Returns:
            LLMResponse with potential function calls
        """
        try:
            # Convert tools to Gemini format if needed
            gemini_tools = self._convert_tools_to_gemini_format(tools)
            
            # Create model with tools
            model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=self.generation_config,
                tools=gemini_tools
            )
            
            # Generate response
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: model.generate_content(prompt)
            )
            
            # Process response (including function calls)
            content = ""
            function_calls = []
            
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if part.text:
                            content += part.text
                        elif hasattr(part, 'function_call') and part.function_call:
                            function_calls.append({
                                "name": part.function_call.name,
                                "arguments": dict(part.function_call.args)
                            })
            
            # Add function calls to response metadata
            metadata = {"function_calls": function_calls} if function_calls else None
            
            return LLMResponse(
                content=content,
                model=self.model_name,
                metadata=metadata
            )
            
        except Exception as e:
            raise Exception(f"Gemini function calling error: {str(e)}")
    
    def _convert_tools_to_gemini_format(self, tools: List[Dict[str, Any]]) -> List[Any]:
        """Convert tool definitions to Gemini format"""
        gemini_tools = []
        
        for tool in tools:
            # Convert OpenAI-style function definition to Gemini format
            if "function" in tool:
                func_def = tool["function"]
                gemini_func = genai.protos.FunctionDeclaration(
                    name=func_def["name"],
                    description=func_def.get("description", ""),
                    parameters=func_def.get("parameters", {})
                )
                gemini_tools.append(genai.protos.Tool(function_declarations=[gemini_func]))
        
        return gemini_tools

