"""
Enhanced Tool Execution Layer for the Enhanced Tool Calling Framework
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum
import traceback

from .base_tool import BaseTool, ToolResult


class RetryStrategy(Enum):
    """Retry strategies for failed tool executions"""
    NONE = "none"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    FIXED_DELAY = "fixed_delay"
    IMMEDIATE = "immediate"


@dataclass
class ExecutionConfig:
    """Configuration for tool execution"""
    timeout_seconds: float = 30.0
    max_retries: int = 3
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    retry_delay: float = 1.0
    enable_logging: bool = True
    parallel_execution: bool = False


@dataclass
class ExecutionResult:
    """Result of tool execution with metadata"""
    tool_result: ToolResult
    execution_config: ExecutionConfig
    attempt_count: int
    total_execution_time: float
    retry_history: List[str]


class EnhancedToolExecutionLayer:
    """
    Enhanced tool execution layer that provides robust execution, error handling,
    retry mechanisms, and result formatting for the Enhanced Tool Calling Framework.
    """
    
    def __init__(self, default_config: Optional[ExecutionConfig] = None):
        """
        Initialize the execution layer.
        
        Args:
            default_config: Default execution configuration
        """
        self.default_config = default_config or ExecutionConfig()
        self.logger = logging.getLogger(__name__)
        self.execution_hooks: Dict[str, List[Callable]] = {
            "before_execution": [],
            "after_execution": [],
            "on_error": [],
            "on_retry": []
        }
    
    def add_hook(self, hook_type: str, hook_function: Callable):
        """
        Add an execution hook.
        
        Args:
            hook_type: Type of hook (before_execution, after_execution, on_error, on_retry)
            hook_function: Function to call
        """
        if hook_type in self.execution_hooks:
            self.execution_hooks[hook_type].append(hook_function)
        else:
            raise ValueError(f"Invalid hook type: {hook_type}")
    
    async def execute_tool(
        self, 
        tool: BaseTool, 
        arguments: Dict[str, Any],
        config: Optional[ExecutionConfig] = None
    ) -> ExecutionResult:
        """
        Execute a single tool with the given arguments.
        
        Args:
            tool: Tool to execute
            arguments: Arguments for the tool
            config: Execution configuration (uses default if not provided)
            
        Returns:
            ExecutionResult with the tool result and metadata
        """
        exec_config = config or self.default_config
        start_time = time.time()
        retry_history = []
        
        # Run before_execution hooks
        await self._run_hooks("before_execution", tool, arguments, exec_config)
        
        for attempt in range(exec_config.max_retries + 1):
            try:
                if exec_config.enable_logging:
                    self.logger.info(f"Executing tool '{tool.name}' (attempt {attempt + 1})")
                
                # Validate arguments
                validated_args = tool.validate_arguments(arguments)
                
                # Execute with timeout
                tool_result = await asyncio.wait_for(
                    tool.execute(validated_args),
                    timeout=exec_config.timeout_seconds
                )
                
                # Success - run after_execution hooks
                await self._run_hooks("after_execution", tool, arguments, exec_config, tool_result)
                
                total_time = time.time() - start_time
                
                return ExecutionResult(
                    tool_result=tool_result,
                    execution_config=exec_config,
                    attempt_count=attempt + 1,
                    total_execution_time=total_time,
                    retry_history=retry_history
                )
                
            except Exception as e:
                error_msg = f"Attempt {attempt + 1} failed: {str(e)}"
                retry_history.append(error_msg)
                
                if exec_config.enable_logging:
                    self.logger.warning(error_msg)
                
                # Run error hooks
                await self._run_hooks("on_error", tool, arguments, exec_config, e)
                
                # If this is the last attempt, return error result
                if attempt == exec_config.max_retries:
                    total_time = time.time() - start_time
                    
                    error_result = ToolResult(
                        success=False,
                        error=f"Tool execution failed after {attempt + 1} attempts: {str(e)}",
                        execution_time=total_time,
                        metadata={
                            "exception_type": type(e).__name__,
                            "traceback": traceback.format_exc(),
                            "retry_history": retry_history
                        }
                    )
                    
                    return ExecutionResult(
                        tool_result=error_result,
                        execution_config=exec_config,
                        attempt_count=attempt + 1,
                        total_execution_time=total_time,
                        retry_history=retry_history
                    )
                
                # Calculate retry delay
                delay = self._calculate_retry_delay(exec_config, attempt)
                
                if delay > 0:
                    if exec_config.enable_logging:
                        self.logger.info(f"Retrying in {delay} seconds...")
                    
                    # Run retry hooks
                    await self._run_hooks("on_retry", tool, arguments, exec_config, attempt, delay)
                    
                    await asyncio.sleep(delay)
    
    async def execute_tools_parallel(
        self, 
        tool_executions: List[tuple[BaseTool, Dict[str, Any]]], 
        config: Optional[ExecutionConfig] = None
    ) -> List[ExecutionResult]:
        """
        Execute multiple tools in parallel.
        
        Args:
            tool_executions: List of (tool, arguments) tuples
            config: Execution configuration
            
        Returns:
            List of ExecutionResults in the same order as input
        """
        exec_config = config or self.default_config
        
        if not exec_config.parallel_execution:
            # Execute sequentially if parallel execution is disabled
            results = []
            for tool, args in tool_executions:
                result = await self.execute_tool(tool, args, exec_config)
                results.append(result)
            return results
        
        # Execute in parallel
        tasks = [
            self.execute_tool(tool, args, exec_config)
            for tool, args in tool_executions
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to error results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                tool, args = tool_executions[i]
                error_result = ToolResult(
                    success=False,
                    error=f"Parallel execution failed: {str(result)}",
                    metadata={
                        "exception_type": type(result).__name__,
                        "traceback": traceback.format_exc()
                    }
                )
                processed_results.append(ExecutionResult(
                    tool_result=error_result,
                    execution_config=exec_config,
                    attempt_count=1,
                    total_execution_time=0.0,
                    retry_history=[str(result)]
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    def _calculate_retry_delay(self, config: ExecutionConfig, attempt: int) -> float:
        """Calculate delay before retry based on strategy"""
        if config.retry_strategy == RetryStrategy.NONE:
            return 0.0
        elif config.retry_strategy == RetryStrategy.IMMEDIATE:
            return 0.0
        elif config.retry_strategy == RetryStrategy.FIXED_DELAY:
            return config.retry_delay
        elif config.retry_strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            return config.retry_delay * (2 ** attempt)
        else:
            return config.retry_delay
    
    async def _run_hooks(self, hook_type: str, *args, **kwargs):
        """Run all hooks of a specific type"""
        for hook in self.execution_hooks[hook_type]:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(*args, **kwargs)
                else:
                    hook(*args, **kwargs)
            except Exception as e:
                if self.default_config.enable_logging:
                    self.logger.warning(f"Hook {hook.__name__} failed: {str(e)}")
    
    def format_result_for_llm(self, execution_result: ExecutionResult) -> str:
        """
        Format execution result in a way that's easy for LLMs to understand.
        
        Args:
            execution_result: Result to format
            
        Returns:
            Formatted string representation
        """
        result = execution_result.tool_result
        
        if result.success:
            formatted = f"Tool execution successful.\n"
            formatted += f"Result: {result.data}\n"
            
            if result.execution_time:
                formatted += f"Execution time: {result.execution_time:.2f} seconds\n"
            
            if result.metadata:
                formatted += f"Additional info: {result.metadata}\n"
        else:
            formatted = f"Tool execution failed.\n"
            formatted += f"Error: {result.error}\n"
            
            if execution_result.retry_history:
                formatted += f"Retry attempts: {len(execution_result.retry_history)}\n"
                formatted += f"Retry history: {'; '.join(execution_result.retry_history)}\n"
        
        return formatted.strip()
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """
        Get execution statistics (placeholder for future implementation).
        
        Returns:
            Dictionary with execution statistics
        """
        return {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0.0,
            "most_used_tools": [],
            "error_patterns": []
        }

