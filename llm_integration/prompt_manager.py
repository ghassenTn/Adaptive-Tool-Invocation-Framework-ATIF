"""
Prompt Manager for LLM Integration
"""

import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ToolDescription:
    """Description of a tool for LLM prompts"""
    name: str
    description: str
    parameters: List[Dict[str, Any]]
    examples: Optional[List[str]] = None


@dataclass
class PromptTemplate:
    """Template for generating prompts"""
    system_prompt: str
    user_prompt_template: str
    response_format: str
    language: str = "en"


class PromptManager:
    """Manages prompts for LLM interaction with the tool calling framework"""
    
    def __init__(self, language: str = "en"):
        """
        Initialize prompt manager
        
        Args:
            language: Language for prompts (en, ar)
        """
        self.language = language
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, PromptTemplate]:
        """Load prompt templates for different languages"""
        templates = {}
        
        # English templates
        templates["en"] = PromptTemplate(
            system_prompt="""You are an intelligent assistant that can use tools to help users accomplish tasks. 

Your role is to:
1. Understand what the user wants to do
2. Determine if any available tools can help
3. Generate a clear, natural language query that describes the task

Available tools:
{tools_description}

IMPORTANT INSTRUCTIONS:
- Always respond in JSON format as specified
- If you can help with the user's request using available tools, generate a clear query
- If no tools can help, explain why in the response
- Keep queries natural and descriptive
- Include all necessary information from the user's request

Response format:
{response_format}""",
            
            user_prompt_template="""User request: {user_input}

Please analyze this request and respond in the specified JSON format.""",
            
            response_format="""{
  "can_help": true/false,
  "tool_query": "natural language description of what to do",
  "reasoning": "explanation of your decision",
  "suggested_alternatives": ["alternative suggestions if can't help"]
}"""
        )
        
        # Arabic templates
        templates["ar"] = PromptTemplate(
            system_prompt="""أنت مساعد ذكي يمكنه استخدام الأدوات لمساعدة المستخدمين في إنجاز المهام.

دورك هو:
1. فهم ما يريد المستخدم فعله
2. تحديد ما إذا كانت الأدوات المتاحة يمكنها المساعدة
3. توليد استعلام واضح بلغة طبيعية يصف المهمة

الأدوات المتاحة:
{tools_description}

تعليمات مهمة:
- استجب دائماً بتنسيق JSON كما هو محدد
- إذا كان بإمكانك المساعدة باستخدام الأدوات المتاحة، ولد استعلاماً واضحاً
- إذا لم تتمكن الأدوات من المساعدة، اشرح السبب
- اجعل الاستعلامات طبيعية ووصفية
- ضمن جميع المعلومات الضرورية من طلب المستخدم

تنسيق الاستجابة:
{response_format}""",
            
            user_prompt_template="""طلب المستخدم: {user_input}

يرجى تحليل هذا الطلب والاستجابة بتنسيق JSON المحدد.""",
            
            response_format="""{
  "can_help": true/false,
  "tool_query": "وصف بلغة طبيعية لما يجب فعله",
  "reasoning": "شرح قرارك",
  "suggested_alternatives": ["اقتراحات بديلة إذا لم تتمكن من المساعدة"]
}"""
        )
        
        return templates
    
    def generate_tools_description(self, tools: List[ToolDescription]) -> str:
        """Generate description of available tools for prompts"""
        if self.language == "ar":
            return self._generate_tools_description_ar(tools)
        else:
            return self._generate_tools_description_en(tools)
    
    def _generate_tools_description_en(self, tools: List[ToolDescription]) -> str:
        """Generate English tools description"""
        descriptions = []
        
        for tool in tools:
            desc = f"**{tool.name}**: {tool.description}\n"
            
            if tool.parameters:
                desc += "  Parameters:\n"
                for param in tool.parameters:
                    required = "*" if param.get("required", False) else ""
                    desc += f"    - {param['name']} ({param['type']}){required}: {param['description']}\n"
            
            if tool.examples:
                desc += "  Examples:\n"
                for example in tool.examples:
                    desc += f"    - \"{example}\"\n"
            
            descriptions.append(desc)
        
        return "\n".join(descriptions)
    
    def _generate_tools_description_ar(self, tools: List[ToolDescription]) -> str:
        """Generate Arabic tools description"""
        descriptions = []
        
        for tool in tools:
            desc = f"**{tool.name}**: {tool.description}\n"
            
            if tool.parameters:
                desc += "  المعاملات:\n"
                for param in tool.parameters:
                    required = "*" if param.get("required", False) else ""
                    desc += f"    - {param['name']} ({param['type']}){required}: {param['description']}\n"
            
            if tool.examples:
                desc += "  أمثلة:\n"
                for example in tool.examples:
                    desc += f"    - \"{example}\"\n"
            
            descriptions.append(desc)
        
        return "\n".join(descriptions)
    
    def create_prompt(self, user_input: str, tools: List[ToolDescription]) -> str:
        """Create complete prompt for LLM"""
        template = self.templates[self.language]
        tools_description = self.generate_tools_description(tools)
        
        # Format system prompt
        system_prompt = template.system_prompt.format(
            tools_description=tools_description,
            response_format=template.response_format
        )
        
        # Format user prompt
        user_prompt = template.user_prompt_template.format(
            user_input=user_input
        )
        
        # Combine prompts
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        return full_prompt
    
    def parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response and extract structured data"""
        try:
            # Try to extract JSON from response
            response = response.strip()
            
            # Find JSON block
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx]
                parsed = json.loads(json_str)
                
                # Validate required fields
                required_fields = ["can_help", "reasoning"]
                for field in required_fields:
                    if field not in parsed:
                        raise ValueError(f"Missing required field: {field}")
                
                return parsed
            else:
                raise ValueError("No JSON found in response")
                
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback: try to extract information manually
            return self._fallback_parse(response)
    
    def _fallback_parse(self, response: str) -> Dict[str, Any]:
        """Fallback parsing when JSON parsing fails"""
        response_lower = response.lower()
        
        # Try to determine if LLM thinks it can help
        can_help_indicators = ["yes", "true", "can help", "able to", "يمكنني", "نعم"]
        cannot_help_indicators = ["no", "false", "cannot", "unable", "لا يمكنني", "لا"]
        
        can_help = False
        if any(indicator in response_lower for indicator in can_help_indicators):
            can_help = True
        elif any(indicator in response_lower for indicator in cannot_help_indicators):
            can_help = False
        
        return {
            "can_help": can_help,
            "tool_query": response if can_help else "",
            "reasoning": "Fallback parsing - could not extract structured response",
            "suggested_alternatives": []
        }
    
    def create_tool_description_from_schema(self, tool_schema: Dict[str, Any]) -> ToolDescription:
        """Create ToolDescription from tool schema"""
        parameters = []
        
        if "parameters" in tool_schema and "properties" in tool_schema["parameters"]:
            required_params = tool_schema["parameters"].get("required", [])
            
            for param_name, param_info in tool_schema["parameters"]["properties"].items():
                parameters.append({
                    "name": param_name,
                    "type": param_info.get("type", "unknown"),
                    "description": param_info.get("description", ""),
                    "required": param_name in required_params
                })
        
        return ToolDescription(
            name=tool_schema.get("name", "unknown"),
            description=tool_schema.get("description", ""),
            parameters=parameters
        )
    
    def add_custom_template(self, language: str, template: PromptTemplate):
        """Add custom prompt template for a language"""
        self.templates[language] = template
    
    def set_language(self, language: str):
        """Set the language for prompts"""
        if language not in self.templates:
            raise ValueError(f"Language {language} not supported. Available: {list(self.templates.keys())}")
        self.language = language

