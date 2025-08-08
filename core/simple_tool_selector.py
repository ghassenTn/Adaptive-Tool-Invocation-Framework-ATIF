"""
Simple Tool Selector - Selects the best tool based on keyword matching.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .base_tool import BaseTool


@dataclass
class ToolSelectionResult:
    """Result of tool selection"""
    selected_tool: Optional[BaseTool]
    confidence_score: float
    reasoning: str
    alternatives: List[tuple[BaseTool, float]]  # List of (tool, score) tuples


class SimpleToolSelector:
    """
    Selects the best tool based on keyword matching and predefined patterns.
    This version has no external dependencies.
    """
    
    def __init__(self):
        self.registered_tools: Dict[str, BaseTool] = {}
        self.keyword_patterns: Dict[str, List[str]] = self._load_default_patterns()
        self.tool_categories: Dict[str, str] = {}

    def register_tool(self, tool: BaseTool):
        """
        Registers a single tool.
        """
        self.registered_tools[tool.name] = tool
        # Infer category from tool name or add a default
        self.tool_categories[tool.name] = self._infer_tool_category(tool.name)

    def register_tools(self, tools: List[BaseTool]):
        """
        Registers multiple tools.
        """
        for tool in tools:
            self.register_tool(tool)

    def _load_default_patterns(self) -> Dict[str, List[str]]:
        """
        Loads predefined keyword patterns for common tool categories.
        """
        return {
            "calculator": [
                "add", "sum", "plus", "subtract", "minus", "multiply", "times",
                "divide", "division", "calculate", "compute", "square root", "power",
                "اجمع", "اطرح", "اضرب", "اقسم", "احسب", "جذر تربيعي", "أس"
            ],
            "weather": [
                "weather", "temperature", "forecast", "climate", "طقس", "درجة حرارة", "توقعات"
            ],
            "text_analyzer": [
                "analyze text", "sentiment", "summarize", "count words", "تحليل نص", "مشاعر", "تلخيص"
            ],
            "time_tool": [
                "time", "date", "وقت", "تاريخ"
            ],
            "file_info": [
                "file info", "document info", "حجم ملف", "معلومات ملف"
            ]
        }

    def _infer_tool_category(self, tool_name: str) -> str:
        """
        Infers a category for a tool based on its name.
        """
        tool_name_lower = tool_name.lower()
        for category, patterns in self.keyword_patterns.items():
            if any(pattern in tool_name_lower for pattern in patterns):
                return category
        return "general"

    def _score_tool(self, tool: BaseTool, query: str) -> float:
        """
        Scores a tool based on how well its name, description, and associated
        keywords match the query.
        """
        score = 0.0
        query_lower = query.lower()

        # 1. Direct match with tool name
        if tool.name.lower() in query_lower:
            score += 0.5  # High confidence for direct match

        # 2. Keyword pattern matching
        category = self.tool_categories.get(tool.name, "general")
        if category in self.keyword_patterns:
            for pattern in self.keyword_patterns[category]:
                if pattern in query_lower:
                    score += 0.2  # Moderate confidence for keyword match

        # 3. Description keyword matching
        for word in tool.description.lower().split():
            if word in query_lower:
                score += 0.05  # Low confidence for description match

        # 4. Parameter name matching
        for param in tool.parameters:
            if param.name.lower() in query_lower:
                score += 0.1

        return score

    def select_tool(self, query: str) -> ToolSelectionResult:
        """
        Selects the best tool for a given query.
        """
        if not self.registered_tools:
            return ToolSelectionResult(None, 0.0, "No tools registered.", [])

        scores: Dict[BaseTool, float] = {}
        for tool_name, tool in self.registered_tools.items():
            scores[tool] = self._score_tool(tool, query)

        # Sort tools by score in descending order
        sorted_tools = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        best_tool: Optional[BaseTool] = None
        best_score = 0.0
        reasoning = ""
        alternatives: List[tuple[BaseTool, float]] = []

        if sorted_tools:
            best_tool, best_score = sorted_tools[0]
            alternatives = sorted_tools[1:]

            if best_score > 0.1:  # A low threshold for initial selection
                reasoning = f"Selected '{best_tool.name}' for query: '{query}' | "
                reasoning += f"Confidence score: {best_score:.2f} | "
                
                # Add more detailed reasoning based on what contributed to the score
                query_lower = query.lower()
                if best_tool.name.lower() in query_lower:
                    reasoning += "Tool name directly mentioned in query | "
                
                category = self.tool_categories.get(best_tool.name, "general")
                if category in self.keyword_patterns:
                    matched_patterns = [p for p in self.keyword_patterns[category] if p in query_lower]
                    if matched_patterns:
                        reasoning += f"Matched {category} patterns ({', '.join(matched_patterns)}) | "
                
                matched_desc_words = [w for w in best_tool.description.lower().split() if w in query_lower]
                if matched_desc_words:
                    reasoning += f"Matched description keywords ({', '.join(matched_desc_words)}) | "
                
                matched_param_names = [p.name for p in best_tool.parameters if p.name.lower() in query_lower]
                if matched_param_names:
                    reasoning += f"Matched parameter names ({', '.join(matched_param_names)}) | "
                
                reasoning += "High confidence - strong match found" if best_score > 0.5 else "Very low confidence - no clear match"

            else:
                best_tool = None # If confidence is too low, don't select any tool
                reasoning = "No suitable tool found with sufficient confidence."

        return ToolSelectionResult(best_tool, best_score, reasoning, alternatives)

