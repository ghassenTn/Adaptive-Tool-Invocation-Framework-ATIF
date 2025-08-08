"""
Simple Test script for the Enhanced Tool Calling Framework
(No external dependencies version)
"""

import asyncio
import time
import json
from typing import List, Dict, Any

# Import framework components
from core.simple_unified_api import SimpleUnifiedToolAPI, ToolCallRequest
from core.execution_layer import ExecutionConfig, RetryStrategy
from examples.sample_tools import (
    CalculatorTool, WeatherTool, TextAnalyzerTool, 
    TimeTool, FileInfoTool
)


class SimpleFrameworkTester:
    """Simple tester for the Enhanced Tool Calling Framework"""
    
    def __init__(self):
        """Initialize the tester with the framework and sample tools"""
        self.api = SimpleUnifiedToolAPI(feedback_db_path="simple_test_feedback.db")
        
        # Register sample tools
        self.tools = [
            CalculatorTool(),
            WeatherTool(),
            TextAnalyzerTool(),
            TimeTool(),
            FileInfoTool()
        ]
        
        self.api.register_tools(self.tools)
        
        # Test cases for different scenarios
        self.test_cases = self._create_test_cases()
    
    def _create_test_cases(self) -> List[Dict[str, Any]]:
        """Create comprehensive test cases"""
        return [
            # Calculator tests
            {
                "name": "Simple Addition",
                "query": "add 15 and 25",
                "expected_tool": "calculator",
                "expected_success": True,
                "description": "Test basic addition operation"
            },
            {
                "name": "Division with Clear Parameters",
                "query": "divide 100 by 4",
                "expected_tool": "calculator",
                "expected_success": True,
                "description": "Test division with clear parameter extraction"
            },
            {
                "name": "Square Root",
                "query": "what is the square root of 64",
                "expected_tool": "calculator",
                "expected_success": True,
                "description": "Test square root operation"
            },
            {
                "name": "Multiplication",
                "query": "multiply 7 times 8",
                "expected_tool": "calculator",
                "expected_success": True,
                "description": "Test multiplication operation"
            },
            
            # Weather tests
            {
                "name": "Weather Query",
                "query": "what's the weather in New York",
                "expected_tool": "weather",
                "expected_success": True,
                "description": "Test weather information retrieval"
            },
            {
                "name": "Weather with Units",
                "query": "get weather for London in fahrenheit",
                "expected_tool": "weather",
                "expected_success": True,
                "description": "Test weather with specific units"
            },
            
            # Text analysis tests
            {
                "name": "Text Analysis",
                "query": "analyze this text: The quick brown fox jumps over the lazy dog",
                "expected_tool": "text_analyzer",
                "expected_success": True,
                "description": "Test basic text analysis"
            },
            {
                "name": "Sentiment Analysis",
                "query": "analyze sentiment of: This is a wonderful and amazing product",
                "expected_tool": "text_analyzer",
                "expected_success": True,
                "description": "Test sentiment analysis"
            },
            
            # Time tests
            {
                "name": "Current Time",
                "query": "what time is it now",
                "expected_tool": "time_tool",
                "expected_success": True,
                "description": "Test current time retrieval"
            },
            {
                "name": "Current Date",
                "query": "what is today's date",
                "expected_tool": "time_tool",
                "expected_success": True,
                "description": "Test current date retrieval"
            },
            
            # File info tests
            {
                "name": "File Information",
                "query": "get info about document.pdf",
                "expected_tool": "file_info",
                "expected_success": True,
                "description": "Test file information retrieval"
            },
            
            # Edge cases
            {
                "name": "Ambiguous Query",
                "query": "help me with something",
                "expected_tool": None,
                "expected_success": False,
                "description": "Test handling of ambiguous queries"
            },
            {
                "name": "Missing Parameters",
                "query": "calculate something",
                "expected_tool": "calculator",
                "expected_success": False,
                "description": "Test handling of missing required parameters"
            },
            
            # Complex queries
            {
                "name": "Complex Math",
                "query": "I need to calculate 25 plus 17",
                "expected_tool": "calculator",
                "expected_success": True,
                "description": "Test complex math query"
            },
            {
                "name": "Weather with Location",
                "query": "Can you tell me the weather forecast for Paris",
                "expected_tool": "weather",
                "expected_success": True,
                "description": "Test weather query with natural language"
            }
        ]
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all test cases and return results"""
        print("🚀 Starting Simple Enhanced Tool Calling Framework Tests")
        print("=" * 60)
        
        results = {
            "total_tests": len(self.test_cases),
            "passed": 0,
            "failed": 0,
            "test_results": [],
            "performance_metrics": {},
            "start_time": time.time()
        }
        
        for i, test_case in enumerate(self.test_cases, 1):
            print(f"\n📋 Test {i}/{len(self.test_cases)}: {test_case['name']}")
            print(f"Query: '{test_case['query']}'")
            print(f"Description: {test_case['description']}")
            
            test_result = await self._run_single_test(test_case)
            results["test_results"].append(test_result)
            
            if test_result["passed"]:
                results["passed"] += 1
                print("✅ PASSED")
            else:
                results["failed"] += 1
                print("❌ FAILED")
                print(f"   Reason: {test_result['failure_reason']}")
            
            if test_result["response"]["success"]:
                print(f"   Tool Used: {test_result['response']['tool_used']}")
                print(f"   Confidence: {test_result['response']['confidence_score']:.2f}")
                print(f"   Execution Time: {test_result['response']['execution_time']:.3f}s")
                if test_result['response']['result']:
                    result_str = str(test_result['response']['result'])
                    if len(result_str) > 100:
                        result_str = result_str[:100] + "..."
                    print(f"   Result: {result_str}")
        
        results["end_time"] = time.time()
        results["total_time"] = results["end_time"] - results["start_time"]
        
        # Get framework performance metrics
        try:
            results["performance_metrics"] = self.api.get_performance_metrics()
        except Exception as e:
            results["performance_metrics"] = {"error": str(e)}
        
        self._print_summary(results)
        return results
    
    async def _run_single_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single test case"""
        start_time = time.time()
        
        try:
            request = ToolCallRequest(
                user_query=test_case["query"],
                context={},
                collect_feedback=True
            )
            
            response = await self.api.call_tool(request)
            execution_time = time.time() - start_time
            
            # Determine if test passed
            passed = True
            failure_reason = ""
            
            # Check tool selection
            if test_case["expected_tool"] is not None:
                if response.tool_used != test_case["expected_tool"]:
                    passed = False
                    failure_reason = f"Expected tool '{test_case['expected_tool']}', got '{response.tool_used}'"
            
            # Check success expectation
            if response.success != test_case["expected_success"]:
                passed = False
                if failure_reason:
                    failure_reason += "; "
                failure_reason += f"Expected success={test_case['expected_success']}, got {response.success}"
            
            return {
                "test_name": test_case["name"],
                "passed": passed,
                "failure_reason": failure_reason,
                "response": {
                    "success": response.success,
                    "tool_used": response.tool_used,
                    "confidence_score": response.confidence_score,
                    "execution_time": response.execution_time,
                    "error": response.error,
                    "result": response.result
                },
                "execution_time": execution_time
            }
            
        except Exception as e:
            return {
                "test_name": test_case["name"],
                "passed": False,
                "failure_reason": f"Exception during test: {str(e)}",
                "response": {
                    "success": False,
                    "error": str(e),
                    "tool_used": None,
                    "confidence_score": 0.0,
                    "execution_time": 0.0,
                    "result": None
                },
                "execution_time": time.time() - start_time
            }
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {results['total_tests']}")
        print(f"Passed: {results['passed']} ✅")
        print(f"Failed: {results['failed']} ❌")
        print(f"Success Rate: {(results['passed'] / results['total_tests'] * 100):.1f}%")
        print(f"Total Execution Time: {results['total_time']:.2f}s")
        
        if results["performance_metrics"] and "error" not in results["performance_metrics"]:
            print("\n📈 FRAMEWORK PERFORMANCE METRICS:")
            metrics = results["performance_metrics"]
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.3f}")
                else:
                    print(f"  {key}: {value}")
        
        # Print failed tests details
        failed_tests = [r for r in results["test_results"] if not r["passed"]]
        if failed_tests:
            print("\n❌ FAILED TESTS:")
            for test in failed_tests:
                print(f"  - {test['test_name']}: {test['failure_reason']}")
    
    async def test_performance_under_load(self, num_requests: int = 5) -> Dict[str, Any]:
        """Test framework performance under load"""
        print(f"\n🔥 Performance Test: {num_requests} concurrent requests")
        
        # Create multiple requests
        requests = []
        for i in range(num_requests):
            test_case = self.test_cases[i % len(self.test_cases)]
            requests.append(ToolCallRequest(
                user_query=test_case["query"],
                context={"test_id": i},
                collect_feedback=True
            ))
        
        start_time = time.time()
        responses = await self.api.batch_call_tools(requests)
        end_time = time.time()
        
        # Analyze results
        successful = sum(1 for r in responses if r.success)
        total_time = end_time - start_time
        avg_time_per_request = total_time / num_requests
        
        results = {
            "num_requests": num_requests,
            "successful_requests": successful,
            "failed_requests": num_requests - successful,
            "success_rate": successful / num_requests,
            "total_time": total_time,
            "average_time_per_request": avg_time_per_request,
            "requests_per_second": num_requests / total_time
        }
        
        print(f"Results:")
        print(f"  Successful: {successful}/{num_requests} ({results['success_rate']:.1%})")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"  Avg Time/Request: {avg_time_per_request:.3f}s")
        print(f"  Requests/Second: {results['requests_per_second']:.1f}")
        
        return results
    
    def test_tool_registration(self):
        """Test tool registration functionality"""
        print("\n🔧 Testing Tool Registration")
        
        # Test getting available tools
        tools = self.api.get_available_tools()
        print(f"Registered tools: {len(tools)}")
        
        for tool in tools:
            print(f"  - {tool['name']}: {tool['description']}")
        
        # Test getting tool schema
        for tool_name in ["calculator", "weather"]:
            schema = self.api.get_tool_schema(tool_name)
            if schema:
                print(f"\nSchema for {tool_name}:")
                print(f"  Parameters: {len(schema['parameters']['properties'])}")
                for param_name, param_info in schema['parameters']['properties'].items():
                    required = param_name in schema['parameters'].get('required', [])
                    print(f"    - {param_name} ({param_info['type']}){'*' if required else ''}: {param_info['description']}")
    
    async def test_individual_components(self):
        """Test individual framework components"""
        print("\n🔍 Testing Individual Components")
        
        # Test tool selector
        print("\n1. Tool Selector Test:")
        test_queries = [
            "add 5 and 3",
            "weather in Tokyo", 
            "analyze this text",
            "what time is it"
        ]
        
        for query in test_queries:
            selection_result = self.api.tool_selector.select_tool(query)
            print(f"  Query: '{query}'")
            print(f"  Selected: {selection_result.selected_tool.name if selection_result.selected_tool else 'None'}")
            print(f"  Confidence: {selection_result.confidence_score:.2f}")
            print(f"  Reasoning: {selection_result.reasoning}")
            print()
        
        # Test argument generator
        print("2. Argument Generator Test:")
        calc_tool = next(tool for tool in self.tools if tool.name == "calculator")
        
        test_arg_queries = [
            "add 10 and 20",
            "divide 100 by 5",
            "square root of 16"
        ]
        
        for query in test_arg_queries:
            extraction_result = self.api.argument_generator.generate_arguments(
                query, calc_tool
            )
            print(f"  Query: '{query}'")
            print(f"  Arguments: {extraction_result.arguments}")
            print(f"  Confidence: {extraction_result.confidence_score:.2f}")
            print(f"  Missing: {extraction_result.missing_required}")
            print()


async def main():
    """Main test function"""
    print("Enhanced Tool Calling Framework - Simple Test Suite")
    print("=" * 60)
    
    # Initialize tester
    tester = SimpleFrameworkTester()
    
    # Test tool registration
    tester.test_tool_registration()
    
    # Test individual components
    await tester.test_individual_components()
    
    # Run main test suite
    results = await tester.run_all_tests()
    
    # Performance test
    await tester.test_performance_under_load(3)
    
    # Save results to file
    with open("simple_test_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Test results saved to simple_test_results.json")
    print("🎉 Testing completed!")


if __name__ == "__main__":
    asyncio.run(main())

