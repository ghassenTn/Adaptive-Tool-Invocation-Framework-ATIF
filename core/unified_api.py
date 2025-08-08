"""
Unified API for the Enhanced Tool Calling Framework
"""

import asyncio
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass

from .base_tool import BaseTool
from .tool_selector import IntelligentToolSelector, ToolSelectionResult
from .argument_generator import ToolArgumentGenerator, ArgumentExtractionResult
from .execution_layer import EnhancedToolExecutionLayer, ExecutionResult, ExecutionConfig
from .feedback_loop import FeedbackLoop


@dataclass
class ToolCallRequest:
    """Request for tool calling"""
    user_query: str
    context: Optional[Dict[str, Any]] = None
    execution_config: Optional[ExecutionConfig] = None
    collect_feedback: bool = True


@dataclass
class ToolCallResponse:
    """Response from tool calling"""
    success: bool
    result: Any = None
    error: Optional[str] = None
    tool_used: Optional[str] = None
    confidence_score: float = 0.0
    execution_time: float = 0.0
    feedback_id: Optional[str] = None
    suggestions: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class UnifiedToolAPI:
    """
    Unified API that orchestrates all components of the Enhanced Tool Calling Framework
    to provide a simple interface for tool calling with less powerful LLMs.
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        nlp_model: str = "en_core_web_sm",
        feedback_db_path: str = "feedback.db",
        default_execution_config: Optional[ExecutionConfig] = None
    ):
        """
        Initialize the unified API.
        
        Args:
            embedding_model: Model for semantic similarity in tool selection
            nlp_model: spaCy model for argument extraction
            feedback_db_path: Path to feedback database
            default_execution_config: Default execution configuration
        """
        self.tool_selector = IntelligentToolSelector(embedding_model)
        self.argument_generator = ToolArgumentGenerator(nlp_model)
        self.execution_layer = EnhancedToolExecutionLayer(default_execution_config)
        self.feedback_loop = FeedbackLoop(feedback_db_path)
        
        # Track registered tools
        self.tools: Dict[str, BaseTool] = {}
    
    def register_tool(self, tool: BaseTool):
        """
        Register a tool with the framework.
        
        Args:
            tool: Tool to register
        """
        self.tools[tool.name] = tool
        self.tool_selector.register_tool(tool)
    
    def register_tools(self, tools: List[BaseTool]):
        """
        Register multiple tools with the framework.
        
        Args:
            tools: List of tools to register
        """
        for tool in tools:
            self.tools[tool.name] = tool
        self.tool_selector.register_tools(tools)
    
    async def call_tool(self, request: ToolCallRequest) -> ToolCallResponse:
        """
        Main method to call a tool based on user query.
        
        Args:
            request: Tool call request
            
        Returns:
            Tool call response
        """
        try:
            # Step 1: Select appropriate tool
            selection_result = self.tool_selector.select_tool(
                request.user_query, 
                request.context
            )
            
            if not selection_result.selected_tool:
                return ToolCallResponse(
                    success=False,
                    error="No suitable tool found for the query",
                    confidence_score=selection_result.confidence_score,
                    suggestions=self._generate_tool_suggestions()
                )
            
            selected_tool = selection_result.selected_tool
            
            # Record tool selection feedback
            if request.collect_feedback:
                selection_feedback_id = self.feedback_loop.record_tool_selection(
                    request.user_query,
                    selection_result,
                    context=request.context
                )
            
            # Step 2: Generate arguments for the selected tool
            extraction_result = self.argument_generator.generate_arguments(
                request.user_query,
                selected_tool,
                request.context
            )
            
            # Record argument extraction feedback
            if request.collect_feedback:
                extraction_feedback_id = self.feedback_loop.record_argument_extraction(
                    request.user_query,
                    selected_tool,
                    extraction_result,
                    context=request.context
                )
            
            # Check if required arguments are missing
            if extraction_result.missing_required:
                suggestions = self.argument_generator.suggest_missing_arguments(
                    selected_tool,
                    extraction_result.missing_required
                )
                
                return ToolCallResponse(
                    success=False,
                    error=f"Missing required arguments: {', '.join(extraction_result.missing_required)}",
                    tool_used=selected_tool.name,
                    confidence_score=extraction_result.confidence_score,
                    suggestions=list(suggestions.values()),
                    metadata={
                        "missing_arguments": extraction_result.missing_required,
                        "extraction_details": extraction_result.extraction_details
                    }
                )
            
            # Step 3: Execute the tool
            execution_result = await self.execution_layer.execute_tool(
                selected_tool,
                extraction_result.arguments,
                request.execution_config
            )
            
            # Record execution feedback
            if request.collect_feedback:
                execution_feedback_id = self.feedback_loop.record_execution_result(
                    request.user_query,
                    selected_tool,
                    execution_result,
                    context=request.context
                )
            
            # Step 4: Format and return response
            if execution_result.tool_result.success:
                return ToolCallResponse(
                    success=True,
                    result=execution_result.tool_result.data,
                    tool_used=selected_tool.name,
                    confidence_score=min(selection_result.confidence_score, extraction_result.confidence_score),
                    execution_time=execution_result.total_execution_time,
                    feedback_id=execution_feedback_id if request.collect_feedback else None,
                    metadata={
                        "selection_reasoning": selection_result.reasoning,
                        "extraction_details": extraction_result.extraction_details,
                        "attempt_count": execution_result.attempt_count
                    }
                )
            else:
                return ToolCallResponse(
                    success=False,
                    error=execution_result.tool_result.error,
                    tool_used=selected_tool.name,
                    confidence_score=min(selection_result.confidence_score, extraction_result.confidence_score),
                    execution_time=execution_result.total_execution_time,
                    feedback_id=execution_feedback_id if request.collect_feedback else None,
                    metadata={
                        "retry_history": execution_result.retry_history,
                        "attempt_count": execution_result.attempt_count
                    }
                )
        
        except Exception as e:
            return ToolCallResponse(
                success=False,
                error=f"Framework error: {str(e)}",
                metadata={"exception_type": type(e).__name__}
            )
    
    async def call_tool_by_name(
        self, 
        tool_name: str, 
        arguments: Dict[str, Any],
        execution_config: Optional[ExecutionConfig] = None
    ) -> ToolCallResponse:
        """
        Call a specific tool by name with provided arguments.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Arguments for the tool
            execution_config: Execution configuration
            
        Returns:
            Tool call response
        """
        if tool_name not in self.tools:
            return ToolCallResponse(
                success=False,
                error=f"Tool '{tool_name}' not found",
                suggestions=self._generate_tool_suggestions()
            )
        
        tool = self.tools[tool_name]
        
        try:
            execution_result = await self.execution_layer.execute_tool(
                tool,
                arguments,
                execution_config
            )
            
            if execution_result.tool_result.success:
                return ToolCallResponse(
                    success=True,
                    result=execution_result.tool_result.data,
                    tool_used=tool_name,
                    confidence_score=1.0,  # Direct call has full confidence
                    execution_time=execution_result.total_execution_time
                )
            else:
                return ToolCallResponse(
                    success=False,
                    error=execution_result.tool_result.error,
                    tool_used=tool_name,
                    execution_time=execution_result.total_execution_time,
                    metadata={
                        "retry_history": execution_result.retry_history,
                        "attempt_count": execution_result.attempt_count
                    }
                )
        
        except Exception as e:
            return ToolCallResponse(
                success=False,
                error=f"Tool execution error: {str(e)}",
                tool_used=tool_name,
                metadata={"exception_type": type(e).__name__}
            )
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """
        Get information about all available tools.
        
        Returns:
            List of tool information dictionaries
        """
        return self.tool_selector.list_tools()
    
    def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the schema for a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool schema or None if tool not found
        """
        if tool_name in self.tools:
            return self.tools[tool_name].get_schema()
        return None
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the framework.
        
        Returns:
            Performance metrics dictionary
        """
        metrics = self.feedback_loop.get_performance_metrics()
        return {
            "tool_selection_accuracy": metrics.tool_selection_accuracy,
            "argument_extraction_accuracy": metrics.argument_extraction_accuracy,
            "execution_success_rate": metrics.execution_success_rate,
            "user_satisfaction_score": metrics.user_satisfaction_score,
            "average_response_time": metrics.average_response_time,
            "total_interactions": metrics.total_interactions
        }
    
    def get_improvement_suggestions(self) -> List[Dict[str, Any]]:
        """
        Get suggestions for improving framework performance.
        
        Returns:
            List of improvement suggestions
        """
        return self.feedback_loop.get_improvement_suggestions()
    
    def provide_feedback(
        self,
        feedback_id: str,
        feedback_type: str,
        feedback_data: Dict[str, Any]
    ):
        """
        Provide feedback for a specific interaction.
        
        Args:
            feedback_id: ID of the feedback entry
            feedback_type: Type of feedback
            feedback_data: Feedback data
        """
        # This would update the existing feedback entry
        # Implementation depends on specific feedback requirements
        pass
    
    def _generate_tool_suggestions(self) -> List[str]:
        """Generate suggestions for available tools"""
        if not self.tools:
            return ["No tools are currently registered"]
        
        suggestions = []
        for tool_name, tool in self.tools.items():
            suggestions.append(f"'{tool_name}': {tool.description}")
        
        return suggestions[:5]  # Limit to top 5 suggestions
    
    async def batch_call_tools(
        self, 
        requests: List[ToolCallRequest]
    ) -> List[ToolCallResponse]:
        """
        Call multiple tools in batch (parallel execution).
        
        Args:
            requests: List of tool call requests
            
        Returns:
            List of tool call responses
        """
        tasks = [self.call_tool(request) for request in requests]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error responses
        processed_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                processed_responses.append(ToolCallResponse(
                    success=False,
                    error=f"Batch execution error: {str(response)}",
                    metadata={"exception_type": type(response).__name__}
                ))
            else:
                processed_responses.append(response)
        
        return processed_responses

