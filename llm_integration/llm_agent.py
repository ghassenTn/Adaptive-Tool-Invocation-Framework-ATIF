"""
LLM Agent that integrates with Enhanced Tool Calling Framework
"""

import asyncio
import time
import json
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime

from .llm_wrapper import BaseLLMWrapper, LLMResponse
from .prompt_manager import PromptManager, ToolDescription
from core.simple_unified_api import SimpleUnifiedToolAPI, ToolCallRequest, ToolCallResponse


@dataclass
class ConversationTurn:
    """Single turn in conversation"""
    user_input: str
    agent_response: str
    tool_used: Optional[str] = None
    tool_result: Any = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class AgentConfig:
    """Configuration for LLM Agent"""
    max_conversation_length: int = 10
    tool_call_timeout: float = 30.0
    enable_conversation_memory: bool = True
    enable_tool_feedback: bool = True
    language: str = "en"
    llm_temperature: float = 0.7
    max_retries: int = 3


@dataclass
class AgentResponse:
    """Response from LLM Agent"""
    message: str
    tool_used: Optional[str] = None
    tool_result: Any = None
    success: bool = True
    error: Optional[str] = None
    reasoning: Optional[str] = None
    execution_time: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


class LLMAgent:
    """
    LLM Agent that combines language model capabilities with the Enhanced Tool Calling Framework
    """
    
    def __init__(
        self,
        llm_wrapper: BaseLLMWrapper,
        tool_api: SimpleUnifiedToolAPI,
        config: Optional[AgentConfig] = None
    ):
        """
        Initialize LLM Agent
        
        Args:
            llm_wrapper: LLM wrapper instance
            tool_api: Tool calling API instance
            config: Agent configuration
        """
        self.llm = llm_wrapper
        self.tool_api = tool_api
        self.config = config or AgentConfig()
        
        # Initialize prompt manager
        self.prompt_manager = PromptManager(language=self.config.language)
        
        # Conversation memory
        self.conversation_history: List[ConversationTurn] = []
        
        # Tool descriptions cache
        self._tool_descriptions: Optional[List[ToolDescription]] = None
    
    async def chat(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        Main chat interface - processes user input and returns response
        
        Args:
            user_input: User's message/request
            context: Optional context information
            
        Returns:
            AgentResponse with the agent's reply
        """
        start_time = time.time()
        
        try:
            # Step 1: Get tool descriptions
            tool_descriptions = await self._get_tool_descriptions()
            
            # Step 2: Create prompt for LLM
            prompt = self._create_conversation_prompt(user_input, tool_descriptions)
            
            # Step 3: Get LLM response
            llm_response = await self._call_llm_with_retry(prompt)
            
            # Debugging: Print raw LLM response content
            print(f"\n--- Raw LLM Response Content ---\n{llm_response.content}\n--- End Raw LLM Response Content ---")

            # Step 4: Parse LLM response
            parsed_response = self.prompt_manager.parse_llm_response(llm_response.content)
            
            # Step 5: Handle tool calling if needed
            if parsed_response.get("can_help", False) and parsed_response.get("tool_query"):
                tool_response = await self._execute_tool_call(
                    parsed_response["tool_query"], 
                    context
                )
                
                if tool_response.success:
                    # Generate final response with tool result
                    final_message = await self._generate_final_response(
                        user_input, 
                        tool_response,
                        parsed_response.get("reasoning", "")
                    )
                    
                    # Update conversation history
                    if self.config.enable_conversation_memory:
                        self._add_to_conversation(
                            user_input, 
                            final_message,
                            tool_response.tool_used,
                            tool_response.result
                        )
                    
                    execution_time = time.time() - start_time
                    
                    return AgentResponse(
                        message=final_message,
                        tool_used=tool_response.tool_used,
                        tool_result=tool_response.result,
                        success=True,
                        reasoning=parsed_response.get("reasoning"),
                        execution_time=execution_time,
                        metadata={
                            "llm_model": self.llm.model_name,
                            "tool_confidence": tool_response.confidence_score,
                            "tool_execution_time": tool_response.execution_time
                        }
                    )
                else:
                    # Tool execution failed
                    error_message = self._generate_error_message(
                        user_input,
                        tool_response.error,
                        parsed_response.get("suggested_alternatives", [])
                    )
                    
                    execution_time = time.time() - start_time
                    
                    return AgentResponse(
                        message=error_message,
                        success=False,
                        error=tool_response.error,
                        reasoning=parsed_response.get("reasoning"),
                        execution_time=execution_time
                    )
            else:
                # Cannot help with available tools
                message = self._generate_cannot_help_message(
                    user_input,
                    parsed_response.get("reasoning", ""),
                    parsed_response.get("suggested_alternatives", [])
                )
                
                execution_time = time.time() - start_time
                
                return AgentResponse(
                    message=message,
                    success=True,
                    reasoning=parsed_response.get("reasoning"),
                    execution_time=execution_time
                )
        
        except Exception as e:
            execution_time = time.time() - start_time
            
            return AgentResponse(
                message=f"I encountered an error while processing your request: {str(e)}",
                success=False,
                error=str(e),
                execution_time=execution_time
            )
    
    async def _get_tool_descriptions(self) -> List[ToolDescription]:
        """Get tool descriptions for prompt generation"""
        if self._tool_descriptions is None:
            tools = self.tool_api.get_available_tools()
            self._tool_descriptions = [
                self.prompt_manager.create_tool_description_from_schema(tool)
                for tool in tools
            ]
        
        return self._tool_descriptions
    
    def _create_conversation_prompt(self, user_input: str, tool_descriptions: List[ToolDescription]) -> str:
        """Create prompt including conversation history if enabled"""
        base_prompt = self.prompt_manager.create_prompt(user_input, tool_descriptions)
        
        if not self.config.enable_conversation_memory or not self.conversation_history:
            return base_prompt
        
        # Add conversation history
        history_text = "\n\nConversation History:\n"
        for turn in self.conversation_history[-3:]:  # Last 3 turns
            history_text += f"User: {turn.user_input}\n"
            history_text += f"Assistant: {turn.agent_response}\n"
            if turn.tool_used:
                history_text += f"(Used tool: {turn.tool_used})\n"
            history_text += "\n"
        
        return base_prompt + history_text
    
    async def _call_llm_with_retry(self, prompt: str) -> LLMResponse:
        """Call LLM with retry logic"""
        last_error = None
        
        for attempt in range(self.config.max_retries):
            try:
                response = await self.llm.generate(
                    prompt,
                    temperature=self.config.llm_temperature
                )
                return response
            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        raise last_error
    
    async def _execute_tool_call(self, tool_query: str, context: Optional[Dict[str, Any]]) -> ToolCallResponse:
        """
        Execute tool call using the framework"""
        request = ToolCallRequest(
            user_query=tool_query,
            context=context or {},
            collect_feedback=self.config.enable_tool_feedback
        )
        
        return await self.tool_api.call_tool(request)
    
    async def _generate_final_response(
        self, 
        user_input: str, 
        tool_response: ToolCallResponse,
        reasoning: str
    ) -> str:
        """Generate final response incorporating tool results"""
        
        # Create prompt for final response generation
        if self.config.language == "ar":
            response_prompt = f"""طلب المستخدم: {user_input}

تم استخدام الأداة: {tool_response.tool_used}
نتيجة الأداة: {tool_response.result}

يرجى تقديم استجابة طبيعية ومفيدة للمستخدم تتضمن النتيجة بطريقة واضحة ومفهومة."""
        else:
            response_prompt = f"""User request: {user_input}

Tool used: {tool_response.tool_used}
Tool result: {tool_response.result}

Please provide a natural and helpful response to the user that incorporates the result in a clear and understandable way."""
        
        try:
            llm_response = await self.llm.generate(
                response_prompt,
                temperature=0.3,  # Lower temperature for more consistent responses
                max_tokens=300
            )
            return llm_response.content.strip()
        except:
            # Fallback response
            if self.config.language == "ar":
                return f"تم تنفيذ طلبك بنجاح باستخدام {tool_response.tool_used}. النتيجة: {tool_response.result}"
            else:
                return f"I successfully completed your request using {tool_response.tool_used}. Result: {tool_response.result}"
    
    def _generate_error_message(
        self, 
        user_input: str, 
        error: str, 
        alternatives: List[str]
    ) -> str:
        """Generate error message when tool execution fails"""
        if self.config.language == "ar":
            message = f"عذراً، واجهت مشكلة في تنفيذ طلبك: {error}"
            if alternatives:
                message += f"\n\nيمكنك تجربة:\n" + "\n".join(f"- {alt}" for alt in alternatives)
        else:
            message = f"I'm sorry, I encountered an issue while processing your request: {error}"
            if alternatives:
                message += f"\n\nYou could try:\n" + "\n".join(f"- {alt}" for alt in alternatives)
        
        return message
    
    def _generate_cannot_help_message(
        self, 
        user_input: str, 
        reasoning: str, 
        alternatives: List[str]
    ) -> str:
        """Generate message when agent cannot help with available tools"""
        if self.config.language == "ar":
            message = f"عذراً، لا أستطيع مساعدتك في هذا الطلب باستخدام الأدوات المتاحة حالياً."
            if reasoning:
                message += f"\n\nالسبب: {reasoning}"
            if alternatives:
                message += f"\n\nيمكنك تجربة:\n" + "\n".join(f"- {alt}" for alt in alternatives)
        else:
            message = f"I'm sorry, I cannot help with this request using the currently available tools."
            if reasoning:
                message += f"\n\nReason: {reasoning}"
            if alternatives:
                message += f"\n\nYou could try:\n" + "\n".join(f"- {alt}" for alt in alternatives)
        
        return message
    
    def _add_to_conversation(
        self, 
        user_input: str, 
        agent_response: str, 
        tool_used: Optional[str] = None,
        tool_result: Any = None
    ):
        """Add turn to conversation history"""
        turn = ConversationTurn(
            user_input=user_input,
            agent_response=agent_response,
            tool_used=tool_used,
            tool_result=tool_result
        )
        
        self.conversation_history.append(turn)
        
        # Trim history if too long
        if len(self.conversation_history) > self.config.max_conversation_length:
            self.conversation_history = self.conversation_history[-self.config.max_conversation_length:]
    
    def clear_conversation(self):
        """Clear conversation history"""
        self.conversation_history.clear()
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of conversation"""
        return {
            "total_turns": len(self.conversation_history),
            "tools_used": list(set(turn.tool_used for turn in self.conversation_history if turn.tool_used)),
            "start_time": self.conversation_history[0].timestamp if self.conversation_history else None,
            "last_activity": self.conversation_history[-1].timestamp if self.conversation_history else None
        }
    
    def update_config(self, **kwargs):
        """Update agent configuration"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Update prompt manager language if changed
        if "language" in kwargs:
            self.prompt_manager.set_language(kwargs["language"])
    
    def refresh_tools(self):
        """Refresh tool descriptions cache"""
        self._tool_descriptions = None



