"""
Feedback Loop and Refinement for the Enhanced Tool Calling Framework
"""

import json
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
import threading
from pathlib import Path

from .base_tool import BaseTool
from .simple_tool_selector import ToolSelectionResult
from .simple_argument_generator import ArgumentExtractionResult
from .execution_layer import ExecutionResult


class FeedbackType(Enum):
    """Types of feedback that can be collected"""
    TOOL_SELECTION = "tool_selection"
    ARGUMENT_EXTRACTION = "argument_extraction"
    EXECUTION_RESULT = "execution_result"
    USER_SATISFACTION = "user_satisfaction"


@dataclass
class FeedbackEntry:
    """Single feedback entry"""
    id: str
    timestamp: float
    feedback_type: FeedbackType
    user_query: str
    context: Dict[str, Any]
    system_decision: Dict[str, Any]
    user_feedback: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class PerformanceMetrics:
    """Performance metrics for the framework"""
    tool_selection_accuracy: float
    argument_extraction_accuracy: float
    execution_success_rate: float
    user_satisfaction_score: float
    average_response_time: float
    total_interactions: int


class FeedbackLoop:
    """
    Feedback loop system that collects performance data and user feedback
    to continuously improve the tool calling framework.
    """
    
    def __init__(self, db_path: str = "feedback.db"):
        """
        Initialize the feedback loop.
        
        Args:
            db_path: Path to SQLite database for storing feedback
        """
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_database()
        
        # In-memory cache for recent feedback
        self.recent_feedback: List[FeedbackEntry] = []
        self.max_cache_size = 1000
    
    def _init_database(self):
        """Initialize the SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY,
                    timestamp REAL,
                    feedback_type TEXT,
                    user_query TEXT,
                    context TEXT,
                    system_decision TEXT,
                    user_feedback TEXT,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON feedback(timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_type ON feedback(feedback_type)
            """)
    
    def record_tool_selection(
        self, 
        user_query: str, 
        selection_result: ToolSelectionResult,
        user_feedback: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Record feedback about tool selection.
        
        Args:
            user_query: Original user query
            selection_result: Result from tool selector
            user_feedback: Optional user feedback about the selection
            context: Optional context information
            
        Returns:
            Feedback entry ID
        """
        system_decision = {
            "selected_tool": selection_result.selected_tool.name if selection_result.selected_tool else None,
            "confidence_score": selection_result.confidence_score,
            "reasoning": selection_result.reasoning,
            "alternatives": [
                {"tool": alt[0].name, "score": alt[1]} 
                for alt in selection_result.alternatives
            ]
        }
        
        return self._record_feedback(
            feedback_type=FeedbackType.TOOL_SELECTION,
            user_query=user_query,
            context=context or {},
            system_decision=system_decision,
            user_feedback=user_feedback or {}
        )
    
    def record_argument_extraction(
        self,
        user_query: str,
        tool: BaseTool,
        extraction_result: ArgumentExtractionResult,
        user_feedback: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Record feedback about argument extraction.
        
        Args:
            user_query: Original user query
            tool: Tool for which arguments were extracted
            extraction_result: Result from argument generator
            user_feedback: Optional user feedback about the extraction
            context: Optional context information
            
        Returns:
            Feedback entry ID
        """
        system_decision = {
            "tool_name": tool.name,
            "extracted_arguments": extraction_result.arguments,
            "confidence_score": extraction_result.confidence_score,
            "missing_required": extraction_result.missing_required,
            "extraction_details": extraction_result.extraction_details
        }
        
        return self._record_feedback(
            feedback_type=FeedbackType.ARGUMENT_EXTRACTION,
            user_query=user_query,
            context=context or {},
            system_decision=system_decision,
            user_feedback=user_feedback or {}
        )
    
    def record_execution_result(
        self,
        user_query: str,
        tool: BaseTool,
        execution_result: ExecutionResult,
        user_feedback: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Record feedback about tool execution.
        
        Args:
            user_query: Original user query
            tool: Tool that was executed
            execution_result: Result from execution layer
            user_feedback: Optional user feedback about the execution
            context: Optional context information
            
        Returns:
            Feedback entry ID
        """
        system_decision = {
            "tool_name": tool.name,
            "success": execution_result.tool_result.success,
            "execution_time": execution_result.total_execution_time,
            "attempt_count": execution_result.attempt_count,
            "error": execution_result.tool_result.error,
            "retry_history": execution_result.retry_history
        }
        
        return self._record_feedback(
            feedback_type=FeedbackType.EXECUTION_RESULT,
            user_query=user_query,
            context=context or {},
            system_decision=system_decision,
            user_feedback=user_feedback or {}
        )
    
    def record_user_satisfaction(
        self,
        user_query: str,
        satisfaction_score: float,
        comments: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Record overall user satisfaction feedback.
        
        Args:
            user_query: Original user query
            satisfaction_score: Score from 0.0 to 1.0
            comments: Optional user comments
            context: Optional context information
            
        Returns:
            Feedback entry ID
        """
        user_feedback = {
            "satisfaction_score": satisfaction_score,
            "comments": comments or ""
        }
        
        return self._record_feedback(
            feedback_type=FeedbackType.USER_SATISFACTION,
            user_query=user_query,
            context=context or {},
            system_decision={},
            user_feedback=user_feedback
        )
    
    def _record_feedback(
        self,
        feedback_type: FeedbackType,
        user_query: str,
        context: Dict[str, Any],
        system_decision: Dict[str, Any],
        user_feedback: Dict[str, Any]
    ) -> str:
        """Internal method to record feedback"""
        entry_id = f"{feedback_type.value}_{int(time.time() * 1000000)}"
        timestamp = time.time()
        
        entry = FeedbackEntry(
            id=entry_id,
            timestamp=timestamp,
            feedback_type=feedback_type,
            user_query=user_query,
            context=context,
            system_decision=system_decision,
            user_feedback=user_feedback,
            metadata={}
        )
        
        # Store in database
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO feedback 
                    (id, timestamp, feedback_type, user_query, context, 
                     system_decision, user_feedback, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry.id,
                    entry.timestamp,
                    entry.feedback_type.value,
                    entry.user_query,
                    json.dumps(entry.context),
                    json.dumps(entry.system_decision),
                    json.dumps(entry.user_feedback),
                    json.dumps(entry.metadata)
                ))
            
            # Add to cache
            self.recent_feedback.append(entry)
            if len(self.recent_feedback) > self.max_cache_size:
                self.recent_feedback.pop(0)
        
        return entry_id
    
    def get_performance_metrics(
        self, 
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> PerformanceMetrics:
        """
        Calculate performance metrics for a given time period.
        
        Args:
            start_time: Start timestamp (None for all time)
            end_time: End timestamp (None for current time)
            
        Returns:
            PerformanceMetrics object
        """
        with sqlite3.connect(self.db_path) as conn:
            # Build time filter
            time_filter = ""
            params = []
            if start_time is not None:
                time_filter += " AND timestamp >= ?"
                params.append(start_time)
            if end_time is not None:
                time_filter += " AND timestamp <= ?"
                params.append(end_time)
            
            # Tool selection accuracy
            cursor = conn.execute(f"""
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN json_extract(user_feedback, '$.correct') = 1 THEN 1 ELSE 0 END) as correct
                FROM feedback 
                WHERE feedback_type = 'tool_selection' {time_filter}
            """, params)
            
            row = cursor.fetchone()
            tool_selection_total = row[0] if row[0] else 0
            tool_selection_correct = row[1] if row[1] else 0
            tool_selection_accuracy = tool_selection_correct / tool_selection_total if tool_selection_total > 0 else 0.0
            
            # Argument extraction accuracy
            cursor = conn.execute(f"""
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN json_extract(user_feedback, '$.correct') = 1 THEN 1 ELSE 0 END) as correct
                FROM feedback 
                WHERE feedback_type = 'argument_extraction' {time_filter}
            """, params)
            
            row = cursor.fetchone()
            arg_extraction_total = row[0] if row[0] else 0
            arg_extraction_correct = row[1] if row[1] else 0
            arg_extraction_accuracy = arg_extraction_correct / arg_extraction_total if arg_extraction_total > 0 else 0.0
            
            # Execution success rate
            cursor = conn.execute(f"""
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN json_extract(system_decision, '$.success') = 1 THEN 1 ELSE 0 END) as successful
                FROM feedback 
                WHERE feedback_type = 'execution_result' {time_filter}
            """, params)
            
            row = cursor.fetchone()
            execution_total = row[0] if row[0] else 0
            execution_successful = row[1] if row[1] else 0
            execution_success_rate = execution_successful / execution_total if execution_total > 0 else 0.0
            
            # User satisfaction
            cursor = conn.execute(f"""
                SELECT AVG(CAST(json_extract(user_feedback, '$.satisfaction_score') AS REAL)) as avg_satisfaction
                FROM feedback 
                WHERE feedback_type = 'user_satisfaction' {time_filter}
            """, params)
            
            row = cursor.fetchone()
            user_satisfaction = row[0] if row[0] else 0.0
            
            # Average response time
            cursor = conn.execute(f"""
                SELECT AVG(CAST(json_extract(system_decision, '$.execution_time') AS REAL)) as avg_time
                FROM feedback 
                WHERE feedback_type = 'execution_result' {time_filter}
            """, params)
            
            row = cursor.fetchone()
            avg_response_time = row[0] if row[0] else 0.0
            
            # Total interactions
            cursor = conn.execute(f"""
                SELECT COUNT(DISTINCT user_query) as total
                FROM feedback 
                WHERE 1=1 {time_filter}
            """, params)
            
            row = cursor.fetchone()
            total_interactions = row[0] if row[0] else 0
        
        return PerformanceMetrics(
            tool_selection_accuracy=tool_selection_accuracy,
            argument_extraction_accuracy=arg_extraction_accuracy,
            execution_success_rate=execution_success_rate,
            user_satisfaction_score=user_satisfaction,
            average_response_time=avg_response_time,
            total_interactions=total_interactions
        )
    
    def get_improvement_suggestions(self) -> List[Dict[str, Any]]:
        """
        Analyze feedback data to generate improvement suggestions.
        
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        metrics = self.get_performance_metrics()
        
        # Tool selection improvements
        if metrics.tool_selection_accuracy < 0.8:
            suggestions.append({
                "component": "tool_selector",
                "issue": "Low tool selection accuracy",
                "current_score": metrics.tool_selection_accuracy,
                "suggestion": "Consider retraining the embedding model or updating intent patterns",
                "priority": "high" if metrics.tool_selection_accuracy < 0.6 else "medium"
            })
        
        # Argument extraction improvements
        if metrics.argument_extraction_accuracy < 0.7:
            suggestions.append({
                "component": "argument_generator",
                "issue": "Low argument extraction accuracy",
                "current_score": metrics.argument_extraction_accuracy,
                "suggestion": "Improve NLP patterns or add more semantic extraction rules",
                "priority": "high" if metrics.argument_extraction_accuracy < 0.5 else "medium"
            })
        
        # Execution improvements
        if metrics.execution_success_rate < 0.9:
            suggestions.append({
                "component": "execution_layer",
                "issue": "Low execution success rate",
                "current_score": metrics.execution_success_rate,
                "suggestion": "Review error handling and retry strategies",
                "priority": "high" if metrics.execution_success_rate < 0.8 else "medium"
            })
        
        # User satisfaction improvements
        if metrics.user_satisfaction_score < 0.7:
            suggestions.append({
                "component": "overall_system",
                "issue": "Low user satisfaction",
                "current_score": metrics.user_satisfaction_score,
                "suggestion": "Analyze user feedback comments for specific pain points",
                "priority": "high"
            })
        
        return suggestions
    
    def export_feedback_data(self, output_path: str, format: str = "json"):
        """
        Export feedback data for external analysis.
        
        Args:
            output_path: Path to save the exported data
            format: Export format ("json" or "csv")
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM feedback ORDER BY timestamp")
            rows = cursor.fetchall()
            
            if format == "json":
                data = []
                for row in rows:
                    entry = {
                        "id": row[0],
                        "timestamp": row[1],
                        "feedback_type": row[2],
                        "user_query": row[3],
                        "context": json.loads(row[4]),
                        "system_decision": json.loads(row[5]),
                        "user_feedback": json.loads(row[6]),
                        "metadata": json.loads(row[7])
                    }
                    data.append(entry)
                
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2)
            
            elif format == "csv":
                import csv
                with open(output_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "id", "timestamp", "feedback_type", "user_query",
                        "context", "system_decision", "user_feedback", "metadata"
                    ])
                    writer.writerows(rows)

