"""
Intelligent Tool Selector for the Enhanced Tool Calling Framework
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .base_tool import BaseTool


@dataclass
class ToolSelectionResult:
    """Result of tool selection process"""
    selected_tool: Optional[BaseTool]
    confidence_score: float
    reasoning: str
    alternatives: List[Tuple[BaseTool, float]]


class IntelligentToolSelector:
    """
    Intelligent tool selector that uses semantic similarity and pattern matching
    to select the most appropriate tool for a given user query.
    
    This component is designed to work effectively with less powerful LLMs by
    reducing the reasoning burden on the model itself.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the tool selector.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.embedding_model = SentenceTransformer(model_name)
        self.tools: List[BaseTool] = []
        self.tool_embeddings: Optional[np.ndarray] = None
        self.intent_patterns = self._load_intent_patterns()
    
    def _load_intent_patterns(self) -> Dict[str, List[str]]:
        """Load predefined intent patterns for common tool categories"""
        return {
            "search": [
                r"search for",
                r"find",
                r"look up",
                r"what is",
                r"who is",
                r"where is",
                r"when is",
                r"how to find"
            ],
            "calculation": [
                r"calculate",
                r"compute",
                r"add",
                r"subtract",
                r"multiply",
                r"divide",
                r"what is \d+",
                r"math",
                r"equation"
            ],
            "weather": [
                r"weather",
                r"temperature",
                r"forecast",
                r"rain",
                r"sunny",
                r"cloudy"
            ],
            "time": [
                r"time",
                r"date",
                r"today",
                r"tomorrow",
                r"yesterday",
                r"current time"
            ],
            "file": [
                r"read file",
                r"write file",
                r"save",
                r"load",
                r"download",
                r"upload"
            ]
        }
    
    def register_tool(self, tool: BaseTool):
        """
        Register a new tool with the selector.
        
        Args:
            tool: Tool to register
        """
        self.tools.append(tool)
        self._update_embeddings()
    
    def register_tools(self, tools: List[BaseTool]):
        """
        Register multiple tools at once.
        
        Args:
            tools: List of tools to register
        """
        self.tools.extend(tools)
        self._update_embeddings()
    
    def _update_embeddings(self):
        """Update tool embeddings when tools are added"""
        if not self.tools:
            self.tool_embeddings = None
            return
        
        # Create embeddings for tool descriptions
        descriptions = [tool.description for tool in self.tools]
        self.tool_embeddings = self.embedding_model.encode(descriptions)
    
    def select_tool(self, user_query: str, context: Optional[Dict[str, Any]] = None) -> ToolSelectionResult:
        """
        Select the most appropriate tool for the given user query.
        
        Args:
            user_query: User's natural language query
            context: Optional context information
            
        Returns:
            ToolSelectionResult with the selected tool and metadata
        """
        if not self.tools:
            return ToolSelectionResult(
                selected_tool=None,
                confidence_score=0.0,
                reasoning="No tools available",
                alternatives=[]
            )
        
        # Step 1: Pattern-based intent detection
        intent_scores = self._detect_intent(user_query)
        
        # Step 2: Semantic similarity matching
        semantic_scores = self._compute_semantic_similarity(user_query)
        
        # Step 3: Combine scores
        combined_scores = self._combine_scores(intent_scores, semantic_scores)
        
        # Step 4: Select best tool
        best_tool_idx = np.argmax(combined_scores)
        best_score = combined_scores[best_tool_idx]
        
        # Step 5: Generate alternatives
        alternatives = []
        sorted_indices = np.argsort(combined_scores)[::-1]
        for idx in sorted_indices[1:4]:  # Top 3 alternatives
            if combined_scores[idx] > 0.1:  # Minimum threshold
                alternatives.append((self.tools[idx], combined_scores[idx]))
        
        # Step 6: Generate reasoning
        reasoning = self._generate_reasoning(
            user_query, 
            self.tools[best_tool_idx], 
            best_score,
            intent_scores[best_tool_idx],
            semantic_scores[best_tool_idx]
        )
        
        return ToolSelectionResult(
            selected_tool=self.tools[best_tool_idx] if best_score > 0.2 else None,
            confidence_score=best_score,
            reasoning=reasoning,
            alternatives=alternatives
        )
    
    def _detect_intent(self, user_query: str) -> np.ndarray:
        """
        Detect user intent using pattern matching.
        
        Args:
            user_query: User's query
            
        Returns:
            Array of intent scores for each tool
        """
        query_lower = user_query.lower()
        intent_scores = np.zeros(len(self.tools))
        
        for intent_category, patterns in self.intent_patterns.items():
            # Check if any pattern matches
            pattern_match = False
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    pattern_match = True
                    break
            
            if pattern_match:
                # Find tools that might handle this intent
                for i, tool in enumerate(self.tools):
                    tool_desc_lower = tool.description.lower()
                    if intent_category in tool_desc_lower or any(
                        keyword in tool_desc_lower 
                        for keyword in intent_category.split()
                    ):
                        intent_scores[i] += 0.5
        
        return intent_scores
    
    def _compute_semantic_similarity(self, user_query: str) -> np.ndarray:
        """
        Compute semantic similarity between query and tool descriptions.
        
        Args:
            user_query: User's query
            
        Returns:
            Array of similarity scores for each tool
        """
        if self.tool_embeddings is None:
            return np.zeros(len(self.tools))
        
        query_embedding = self.embedding_model.encode([user_query])
        similarities = cosine_similarity(query_embedding, self.tool_embeddings)[0]
        
        return similarities
    
    def _combine_scores(self, intent_scores: np.ndarray, semantic_scores: np.ndarray) -> np.ndarray:
        """
        Combine intent and semantic scores.
        
        Args:
            intent_scores: Pattern-based intent scores
            semantic_scores: Semantic similarity scores
            
        Returns:
            Combined scores
        """
        # Weighted combination: 40% intent, 60% semantic
        combined = 0.4 * intent_scores + 0.6 * semantic_scores
        
        # Normalize to [0, 1]
        if combined.max() > 0:
            combined = combined / combined.max()
        
        return combined
    
    def _generate_reasoning(
        self, 
        user_query: str, 
        selected_tool: BaseTool, 
        total_score: float,
        intent_score: float,
        semantic_score: float
    ) -> str:
        """
        Generate human-readable reasoning for the tool selection.
        
        Args:
            user_query: Original user query
            selected_tool: Selected tool
            total_score: Combined confidence score
            intent_score: Intent-based score
            semantic_score: Semantic similarity score
            
        Returns:
            Reasoning string
        """
        reasoning_parts = []
        
        reasoning_parts.append(f"Selected '{selected_tool.name}' for query: '{user_query}'")
        reasoning_parts.append(f"Overall confidence: {total_score:.2f}")
        
        if intent_score > 0:
            reasoning_parts.append(f"Intent pattern match score: {intent_score:.2f}")
        
        reasoning_parts.append(f"Semantic similarity score: {semantic_score:.2f}")
        
        if total_score > 0.8:
            reasoning_parts.append("High confidence - strong match found")
        elif total_score > 0.5:
            reasoning_parts.append("Medium confidence - reasonable match")
        else:
            reasoning_parts.append("Low confidence - weak match, consider alternatives")
        
        return " | ".join(reasoning_parts)
    
    def get_tool_by_name(self, name: str) -> Optional[BaseTool]:
        """
        Get a tool by its name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool if found, None otherwise
        """
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """
        Get a list of all registered tools with their metadata.
        
        Returns:
            List of tool information dictionaries
        """
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": [
                    {
                        "name": param.name,
                        "type": param.type.value,
                        "required": param.required,
                        "description": param.description
                    }
                    for param in tool.parameters
                ]
            }
            for tool in self.tools
        ]

