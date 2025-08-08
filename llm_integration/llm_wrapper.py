"""
LLM Wrapper for Enhanced Tool Calling Framework
"""

import json
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import openai
import requests


@dataclass
class LLMResponse:
    """Response from LLM"""
    content: str
    usage: Optional[Dict[str, Any]] = None
    model: Optional[str] = None
    finish_reason: Optional[str] = None


class BaseLLMWrapper(ABC):
    """Base class for LLM wrappers"""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.config = kwargs
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response from LLM"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        pass


class OpenAIWrapper(BaseLLMWrapper):
    """Wrapper for OpenAI models (GPT-3.5, GPT-4, etc.)"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: Optional[str] = None, **kwargs):
        super().__init__(model_name, **kwargs)
        
        # Initialize OpenAI client
        if api_key:
            openai.api_key = api_key
        
        self.client = openai.OpenAI()
        
        # Default parameters
        self.default_params = {
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1000),
            "top_p": kwargs.get("top_p", 1.0),
            "frequency_penalty": kwargs.get("frequency_penalty", 0.0),
            "presence_penalty": kwargs.get("presence_penalty", 0.0)
        }
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using OpenAI API"""
        try:
            # Merge parameters
            params = {**self.default_params, **kwargs}
            
            # Prepare messages
            messages = [{"role": "user", "content": prompt}]
            
            # Make API call
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **params
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                usage=response.usage.model_dump() if response.usage else None,
                model=response.model,
                finish_reason=response.choices[0].finish_reason
            )
            
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get OpenAI model information"""
        return {
            "provider": "OpenAI",
            "model": self.model_name,
            "type": "cloud",
            "supports_streaming": True,
            "max_context_length": self._get_context_length()
        }
    
    def _get_context_length(self) -> int:
        """Get context length for different models"""
        context_lengths = {
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384,
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-4-turbo": 128000,
            "gpt-4o": 128000
        }
        return context_lengths.get(self.model_name, 4096)


class OllamaWrapper(BaseLLMWrapper):
    """Wrapper for Ollama local models"""
    
    def __init__(self, model_name: str, base_url: str = "http://localhost:11434", **kwargs):
        super().__init__(model_name, **kwargs)
        self.base_url = base_url.rstrip('/')
        
        # Default parameters
        self.default_params = {
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "top_k": kwargs.get("top_k", 40),
            "num_predict": kwargs.get("num_predict", 1000)
        }
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using Ollama API"""
        try:
            # Merge parameters
            params = {**self.default_params, **kwargs}
            
            # Prepare request
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": params
            }
            
            # Make API call
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=data,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            
            return LLMResponse(
                content=result.get("response", ""),
                model=self.model_name,
                finish_reason="stop" if result.get("done", False) else "length"
            )
            
        except Exception as e:
            raise Exception(f"Ollama API error: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Ollama model information"""
        try:
            # Try to get model info from Ollama
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_info = next((m for m in models if m["name"] == self.model_name), None)
                
                return {
                    "provider": "Ollama",
                    "model": self.model_name,
                    "type": "local",
                    "supports_streaming": True,
                    "size": model_info.get("size", "unknown") if model_info else "unknown",
                    "modified_at": model_info.get("modified_at") if model_info else None
                }
        except:
            pass
        
        return {
            "provider": "Ollama",
            "model": self.model_name,
            "type": "local",
            "supports_streaming": True
        }


class HuggingFaceWrapper(BaseLLMWrapper):
    """Wrapper for Hugging Face models (local or API)"""
    
    def __init__(self, model_name: str, use_api: bool = False, api_token: Optional[str] = None, **kwargs):
        super().__init__(model_name, **kwargs)
        self.use_api = use_api
        self.api_token = api_token
        
        if not use_api:
            # Local model setup
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    **kwargs
                )
            except ImportError:
                raise Exception("transformers library is required for local Hugging Face models")
        else:
            # API setup
            self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
            self.headers = {"Authorization": f"Bearer {api_token}"} if api_token else {}
    
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using Hugging Face model"""
        try:
            if self.use_api:
                return await self._generate_api(prompt, **kwargs)
            else:
                return await self._generate_local(prompt, **kwargs)
        except Exception as e:
            raise Exception(f"Hugging Face error: {str(e)}")
    
    async def _generate_api(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate using Hugging Face API"""
        data = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": kwargs.get("max_tokens", 100),
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
                "do_sample": True
            }
        }
        
        response = requests.post(
            self.api_url,
            headers=self.headers,
            json=data,
            timeout=60
        )
        response.raise_for_status()
        
        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            generated_text = result[0].get("generated_text", "")
            # Remove the original prompt from the response
            content = generated_text[len(prompt):].strip()
        else:
            content = ""
        
        return LLMResponse(
            content=content,
            model=self.model_name
        )
    
    async def _generate_local(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate using local Hugging Face model"""
        # Run in thread to avoid blocking
        loop = asyncio.get_event_loop()
        
        def _generate():
            result = self.pipeline(
                prompt,
                max_new_tokens=kwargs.get("max_tokens", 100),
                temperature=kwargs.get("temperature", 0.7),
                top_p=kwargs.get("top_p", 0.9),
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            return result[0]["generated_text"]
        
        generated_text = await loop.run_in_executor(None, _generate)
        
        # Remove the original prompt from the response
        content = generated_text[len(prompt):].strip()
        
        return LLMResponse(
            content=content,
            model=self.model_name
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Hugging Face model information"""
        return {
            "provider": "Hugging Face",
            "model": self.model_name,
            "type": "api" if self.use_api else "local",
            "supports_streaming": False
        }


class LLMFactory:
    """Factory for creating LLM wrappers"""
    
    @staticmethod
    def create_llm(provider: str, model_name: str, **kwargs) -> BaseLLMWrapper:
        """Create LLM wrapper based on provider"""
        provider = provider.lower()
        
        if provider == "openai":
            return OpenAIWrapper(model_name, **kwargs)
        elif provider == "ollama":
            return OllamaWrapper(model_name, **kwargs)
        elif provider == "huggingface":
            return HuggingFaceWrapper(model_name, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    @staticmethod
    def get_available_providers() -> List[str]:
        """Get list of available providers"""
        return ["openai", "ollama", "huggingface"]

