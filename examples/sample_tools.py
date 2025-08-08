"""
Sample tools for testing the Enhanced Tool Calling Framework
"""

import asyncio
import random
import math
from datetime import datetime
from typing import Dict, Any

from core.base_tool import BaseTool, ToolParameter, ParameterType, ToolResult


class CalculatorTool(BaseTool):
    """Simple calculator tool for basic arithmetic operations"""
    
    def __init__(self):
        parameters = [
            ToolParameter(
                name="operation",
                type=ParameterType.STRING,
                description="The arithmetic operation to perform",
                required=True,
                enum_values=["add", "subtract", "multiply", "divide", "power", "sqrt"]
            ),
            ToolParameter(
                name="a",
                type=ParameterType.FLOAT,
                description="First number",
                required=True
            ),
            ToolParameter(
                name="b",
                type=ParameterType.FLOAT,
                description="Second number (not required for sqrt)",
                required=False
            )
        ]
        
        super().__init__(
            name="calculator",
            description="Performs basic arithmetic operations like addition, subtraction, multiplication, division, power, and square root",
            parameters=parameters
        )
    
    async def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        """Execute the calculator operation"""
        try:
            operation = arguments["operation"]
            a = arguments["a"]
            b = arguments.get("b")
            
            if operation == "add":
                if b is None:
                    raise ValueError("Addition requires two numbers")
                result = a + b
            elif operation == "subtract":
                if b is None:
                    raise ValueError("Subtraction requires two numbers")
                result = a - b
            elif operation == "multiply":
                if b is None:
                    raise ValueError("Multiplication requires two numbers")
                result = a * b
            elif operation == "divide":
                if b is None:
                    raise ValueError("Division requires two numbers")
                if b == 0:
                    raise ValueError("Cannot divide by zero")
                result = a / b
            elif operation == "power":
                if b is None:
                    raise ValueError("Power operation requires two numbers")
                result = a ** b
            elif operation == "sqrt":
                if a < 0:
                    raise ValueError("Cannot take square root of negative number")
                result = math.sqrt(a)
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            return ToolResult(
                success=True,
                data=result,
                metadata={
                    "operation": operation,
                    "inputs": {"a": a, "b": b}
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e)
            )


class WeatherTool(BaseTool):
    """Mock weather tool that simulates weather data"""
    
    def __init__(self):
        parameters = [
            ToolParameter(
                name="location",
                type=ParameterType.STRING,
                description="The city or location to get weather for",
                required=True
            ),
            ToolParameter(
                name="units",
                type=ParameterType.STRING,
                description="Temperature units",
                required=False,
                default="celsius",
                enum_values=["celsius", "fahrenheit"]
            )
        ]
        
        super().__init__(
            name="weather",
            description="Gets current weather information for a specified location",
            parameters=parameters
        )
    
    async def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        """Execute the weather lookup (mock implementation)"""
        try:
            location = arguments["location"]
            units = arguments.get("units", "celsius")
            
            # Simulate API delay
            await asyncio.sleep(0.5)
            
            # Generate mock weather data
            base_temp = random.randint(15, 30)
            if units == "fahrenheit":
                temperature = base_temp * 9/5 + 32
                temp_unit = "°F"
            else:
                temperature = base_temp
                temp_unit = "°C"
            
            conditions = random.choice(["sunny", "cloudy", "rainy", "partly cloudy"])
            humidity = random.randint(30, 80)
            wind_speed = random.randint(5, 25)
            
            weather_data = {
                "location": location,
                "temperature": f"{temperature:.1f}{temp_unit}",
                "conditions": conditions,
                "humidity": f"{humidity}%",
                "wind_speed": f"{wind_speed} km/h",
                "timestamp": datetime.now().isoformat()
            }
            
            return ToolResult(
                success=True,
                data=weather_data,
                metadata={
                    "units": units,
                    "mock_data": True
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e)
            )


class TextAnalyzerTool(BaseTool):
    """Tool for analyzing text properties"""
    
    def __init__(self):
        parameters = [
            ToolParameter(
                name="text",
                type=ParameterType.STRING,
                description="The text to analyze",
                required=True
            ),
            ToolParameter(
                name="analysis_type",
                type=ParameterType.STRING,
                description="Type of analysis to perform",
                required=False,
                default="basic",
                enum_values=["basic", "detailed", "sentiment"]
            )
        ]
        
        super().__init__(
            name="text_analyzer",
            description="Analyzes text for various properties like word count, character count, and basic sentiment",
            parameters=parameters
        )
    
    async def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        """Execute text analysis"""
        try:
            text = arguments["text"]
            analysis_type = arguments.get("analysis_type", "basic")
            
            # Basic analysis
            word_count = len(text.split())
            char_count = len(text)
            char_count_no_spaces = len(text.replace(" ", ""))
            sentence_count = len([s for s in text.split(".") if s.strip()])
            
            result = {
                "word_count": word_count,
                "character_count": char_count,
                "character_count_no_spaces": char_count_no_spaces,
                "sentence_count": sentence_count
            }
            
            if analysis_type in ["detailed", "sentiment"]:
                # Add more detailed analysis
                avg_word_length = sum(len(word) for word in text.split()) / word_count if word_count > 0 else 0
                result["average_word_length"] = round(avg_word_length, 2)
                result["reading_time_minutes"] = round(word_count / 200, 1)  # Assuming 200 WPM
            
            if analysis_type == "sentiment":
                # Simple sentiment analysis (mock)
                positive_words = ["good", "great", "excellent", "amazing", "wonderful", "fantastic"]
                negative_words = ["bad", "terrible", "awful", "horrible", "disappointing"]
                
                text_lower = text.lower()
                positive_count = sum(1 for word in positive_words if word in text_lower)
                negative_count = sum(1 for word in negative_words if word in text_lower)
                
                if positive_count > negative_count:
                    sentiment = "positive"
                elif negative_count > positive_count:
                    sentiment = "negative"
                else:
                    sentiment = "neutral"
                
                result["sentiment"] = {
                    "overall": sentiment,
                    "positive_indicators": positive_count,
                    "negative_indicators": negative_count
                }
            
            return ToolResult(
                success=True,
                data=result,
                metadata={
                    "analysis_type": analysis_type,
                    "text_preview": text[:50] + "..." if len(text) > 50 else text
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e)
            )


class TimeTool(BaseTool):
    """Tool for time-related operations"""
    
    def __init__(self):
        parameters = [
            ToolParameter(
                name="operation",
                type=ParameterType.STRING,
                description="The time operation to perform",
                required=True,
                enum_values=["current_time", "current_date", "timestamp", "format_time"]
            ),
            ToolParameter(
                name="timezone",
                type=ParameterType.STRING,
                description="Timezone for the operation (optional)",
                required=False,
                default="UTC"
            ),
            ToolParameter(
                name="format",
                type=ParameterType.STRING,
                description="Format string for time formatting (only for format_time operation)",
                required=False,
                default="%Y-%m-%d %H:%M:%S"
            )
        ]
        
        super().__init__(
            name="time_tool",
            description="Provides current time, date, timestamps, and time formatting operations",
            parameters=parameters
        )
    
    async def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        """Execute time operation"""
        try:
            operation = arguments["operation"]
            timezone = arguments.get("timezone", "UTC")
            format_str = arguments.get("format", "%Y-%m-%d %H:%M:%S")
            
            now = datetime.now()
            
            if operation == "current_time":
                result = now.strftime("%H:%M:%S")
            elif operation == "current_date":
                result = now.strftime("%Y-%m-%d")
            elif operation == "timestamp":
                result = int(now.timestamp())
            elif operation == "format_time":
                result = now.strftime(format_str)
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            return ToolResult(
                success=True,
                data=result,
                metadata={
                    "operation": operation,
                    "timezone": timezone,
                    "iso_format": now.isoformat()
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e)
            )


class FileInfoTool(BaseTool):
    """Mock tool for file information (simulated)"""
    
    def __init__(self):
        parameters = [
            ToolParameter(
                name="filename",
                type=ParameterType.STRING,
                description="Name of the file to get information about",
                required=True
            ),
            ToolParameter(
                name="info_type",
                type=ParameterType.STRING,
                description="Type of information to retrieve",
                required=False,
                default="basic",
                enum_values=["basic", "detailed", "permissions"]
            )
        ]
        
        super().__init__(
            name="file_info",
            description="Gets information about a file including size, type, and modification date (simulated)",
            parameters=parameters
        )
    
    async def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        """Execute file info lookup (mock implementation)"""
        try:
            filename = arguments["filename"]
            info_type = arguments.get("info_type", "basic")
            
            # Simulate file lookup delay
            await asyncio.sleep(0.2)
            
            # Generate mock file data
            file_extensions = {
                ".txt": "text/plain",
                ".pdf": "application/pdf",
                ".jpg": "image/jpeg",
                ".png": "image/png",
                ".doc": "application/msword",
                ".py": "text/x-python"
            }
            
            # Determine file type from extension
            file_type = "unknown"
            for ext, mime_type in file_extensions.items():
                if filename.lower().endswith(ext):
                    file_type = mime_type
                    break
            
            # Generate mock data
            file_size = random.randint(1024, 1024*1024*10)  # 1KB to 10MB
            modified_date = datetime.now().isoformat()
            
            result = {
                "filename": filename,
                "file_type": file_type,
                "size_bytes": file_size,
                "size_human": self._format_file_size(file_size),
                "modified_date": modified_date
            }
            
            if info_type in ["detailed", "permissions"]:
                result.update({
                    "created_date": modified_date,
                    "is_directory": False,
                    "is_hidden": filename.startswith("."),
                    "extension": filename.split(".")[-1] if "." in filename else ""
                })
            
            if info_type == "permissions":
                result.update({
                    "readable": True,
                    "writable": True,
                    "executable": filename.endswith((".exe", ".sh", ".py"))
                })
            
            return ToolResult(
                success=True,
                data=result,
                metadata={
                    "info_type": info_type,
                    "mock_data": True
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e)
            )
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"

