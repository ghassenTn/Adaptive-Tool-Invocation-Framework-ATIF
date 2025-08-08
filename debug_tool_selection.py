"""
Debug script to understand tool selection issues
"""

from core.simple_tool_selector import SimpleToolSelector
from examples.sample_tools import CalculatorTool, WeatherTool, TextAnalyzerTool, TimeTool, FileInfoTool

def debug_tool_selection():
    """Debug tool selection process"""
    
    # Initialize selector and tools
    selector = SimpleToolSelector()
    tools = [
        CalculatorTool(),
        WeatherTool(), 
        TextAnalyzerTool(),
        TimeTool(),
        FileInfoTool()
    ]
    
    selector.register_tools(tools)
    
    # Test queries
    test_queries = [
        "add 15 and 25",
        "divide 100 by 4", 
        "what is the square root of 64",
        "what's the weather in New York",
        "analyze this text: Hello world",
        "what time is it now"
    ]
    
    print("🔍 Debugging Tool Selection")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 30)
        
        # Get selection result
        result = selector.select_tool(query)
        
        print(f"Selected Tool: {result.selected_tool.name if result.selected_tool else 'None'}")
        print(f"Confidence: {result.confidence_score:.3f}")
        print(f"Reasoning: {result.reasoning}")
        
        # Debug scoring for each tool
        query_lower = query.lower()
        print("\nTool Scores:")
        for tool in tools:
            score = selector._score_tool(tool, query_lower)
            category = selector._get_tool_category(tool)
            print(f"  {tool.name}: {score:.3f} (category: {category})")
            
            # Check pattern matching
            if category and category in selector.keyword_patterns:
                patterns = selector.keyword_patterns[category]
                pattern_score = selector._match_patterns(query_lower, patterns)
                print(f"    Pattern score: {pattern_score:.3f}")
        
        print("\nAlternatives:")
        for alt_tool, alt_score in result.alternatives[:3]:
            print(f"  {alt_tool.name}: {alt_score:.3f}")

if __name__ == "__main__":
    debug_tool_selection()

