"""
Smart Tool Use framework for OpenAgent.

Replaces keyword-based tool selection with LLM-driven intelligence to decide
when to use CommandExecutor, FileManager, SystemInfo, Git tools, and in what order.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum

from openagent.core.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class ToolIntent(Enum):
    """Categories of tool intentions for structured selection."""
    SYSTEM_INFO = "system_info"
    FILE_OPERATIONS = "file_operations"  
    COMMAND_EXECUTION = "command_execution"
    GIT_OPERATIONS = "git_operations"
    CODE_SEARCH = "code_search"
    MULTI_STEP = "multi_step"
    UNKNOWN = "unknown"


@dataclass
class ToolCall:
    """Structured representation of a tool call."""
    tool_name: str
    parameters: Dict[str, Any]
    intent: ToolIntent
    rationale: str
    order: int = 0
    depends_on: Optional[List[str]] = None


@dataclass
class ToolPlan:
    """A complete plan with ordered tool calls."""
    calls: List[ToolCall]
    explanation: str
    estimated_risk: str  # "low", "medium", "high"
    requires_confirmation: bool = False


class SmartToolSelector:
    """
    LLM-driven tool selection that understands context and intent.
    
    Instead of keyword matching, this uses the LLM to understand what the user
    wants to accomplish and selects the best tools in the right order.
    """
    
    def __init__(self, llm=None, available_tools: Optional[Dict[str, BaseTool]] = None):
        self.llm = llm
        self.available_tools = available_tools or {}
        self.tool_schemas = self._generate_tool_schemas()
    
    def _generate_tool_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Generate JSON schemas for available tools."""
        schemas = {}
        for name, tool in self.available_tools.items():
            # Create a basic schema from the tool's description
            schema = {
                "name": name,
                "description": getattr(tool, 'description', 'No description available'),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
            
            # Add specific parameter schemas based on tool type
            if "command" in name.lower() or "executor" in name.lower():
                schema["parameters"]["properties"] = {
                    "command": {"type": "string", "description": "Shell command to execute"},
                    "explain_only": {"type": "boolean", "description": "Only explain, don't execute"}
                }
                schema["parameters"]["required"] = ["command"]
            
            elif "file" in name.lower() or "manager" in name.lower():
                schema["parameters"]["properties"] = {
                    "operation": {"type": "string", "enum": ["read", "write", "list", "search"]},
                    "path": {"type": "string", "description": "File or directory path"},
                    "content": {"type": "string", "description": "Content for write operations"}
                }
                schema["parameters"]["required"] = ["operation", "path"]
            
            elif "git" in name.lower():
                schema["parameters"]["properties"] = {
                    "subcommand": {"type": "string", "enum": ["status", "log", "diff", "branch", "show"]},
                    "args": {"type": "array", "items": {"type": "string"}}
                }
                schema["parameters"]["required"] = ["subcommand"]
            
            elif "grep" in name.lower() or "search" in name.lower():
                schema["parameters"]["properties"] = {
                    "pattern": {"type": "string", "description": "Search pattern"},
                    "path": {"type": "string", "description": "Path to search in"},
                    "flags": {"type": "array", "items": {"type": "string"}}
                }
                schema["parameters"]["required"] = ["pattern"]
            
            elif "system" in name.lower():
                schema["parameters"]["properties"] = {
                    "info_type": {"type": "string", "enum": ["overview", "processes", "network", "disk"]}
                }
            
            schemas[name] = schema
        
        return schemas
    
    async def analyze_intent(self, user_request: str, context: Optional[Dict[str, Any]] = None) -> ToolIntent:
        """Analyze user intent to categorize the type of assistance needed."""
        
        # Prepare context information
        context_str = ""
        if context:
            context_str = f"\nContext: {json.dumps(context, indent=2)}"
        
        prompt = f"""Analyze this user request and classify it into one of these categories:

Categories:
- system_info: User wants system information (CPU, memory, processes, disk usage)
- file_operations: User wants to read, write, or manage files
- command_execution: User wants to run shell commands
- git_operations: User wants git information (status, log, diff, branches)
- code_search: User wants to search for code or text in repositories
- multi_step: Request requires multiple tools in sequence
- unknown: Cannot determine intent

User request: "{user_request}"{context_str}

Respond with only the category name (e.g., "system_info").
"""
        
        # Quick heuristic before calling LLM
        ur = (user_request or "").lower()
        if any(k in ur for k in ["cpu", "memory", "process", "disk", "system info", "uptime"]):
            return ToolIntent.SYSTEM_INFO
        if any(k in ur for k in ["list files", "read file", "write file", "directory", "path", "file", "open", "save"]):
            return ToolIntent.FILE_OPERATIONS
        if any(k in ur for k in ["git ", "commit", "branch", "merge", "status", "diff", "log"]):
            return ToolIntent.GIT_OPERATIONS
        if any(k in ur for k in ["search", "grep", "find pattern", "scan code"]):
            return ToolIntent.CODE_SEARCH
        if any(k in ur for k in ["run", "execute", "command", "shell", "bash", "sh"]):
            return ToolIntent.COMMAND_EXECUTION
        
        try:
            if self.llm:
                response = await self.llm.generate_response(prompt, system_prompt="Intent Classifier")
                intent_str = response.strip().lower()
                
                for intent in ToolIntent:
                    if intent.value in intent_str:
                        return intent
        except Exception as e:
            logger.warning(f"Intent analysis failed: {e}")
        
        return ToolIntent.UNKNOWN
    
    async def create_tool_plan(self, user_request: str, context: Optional[Dict[str, Any]] = None) -> ToolPlan:
        """Create a structured plan with ordered tool calls."""
        
        intent = await self.analyze_intent(user_request, context)
        
        # Prepare available tools info
        tools_info = []
        for name, schema in self.tool_schemas.items():
            tools_info.append(f"- {name}: {schema['description']}")
        
        context_str = ""
        if context:
            context_str = f"\nContext: {json.dumps(context, indent=2)}"
        
        prompt = f"""Create a tool execution plan for this user request.

Available tools:
{chr(10).join(tools_info)}

User request: "{user_request}"{context_str}
Detected intent: {intent.value}

Respond with a JSON object containing:
{{
    "explanation": "Brief explanation of what will be done",
    "estimated_risk": "low|medium|high",
    "requires_confirmation": true|false,
    "calls": [
        {{
            "tool_name": "exact_tool_name",
            "parameters": {{"key": "value"}},
            "rationale": "Why this tool is needed",
            "order": 1
        }}
    ]
}}

Rules:
1. Only use tools that are actually available
2. Order calls logically (information gathering before actions)
3. Keep explanations concise but informative
4. Mark as high risk if executing potentially dangerous commands
5. Require confirmation for destructive operations
"""
        
        try:
            if self.llm:
                plan_data: Optional[Dict[str, Any]] = None
                # Prefer a structured JSON generation method if available
                generate_json = getattr(self.llm, "generate_json", None)
                if callable(generate_json):
                    plan_data = await generate_json({
                        "prompt": prompt,
                        "schema_hint": "tool_plan_v1"
                    })
                    # Some implementations may wrap under a key
                    if isinstance(plan_data, dict) and "plan" in plan_data and not plan_data.get("calls"):
                        plan_data = plan_data["plan"]
                else:
                    response = await self.llm.generate_response(prompt, system_prompt="Tool Planner")
                    # Extract JSON from response
                    start = response.find('{')
                    end = response.rfind('}')
                    if start != -1 and end != -1:
                        plan_data = json.loads(response[start:end+1])
                
                if isinstance(plan_data, dict):
                    # Convert to ToolPlan
                    tool_calls = []
                    for i, call_data in enumerate(plan_data.get('calls', [])):
                        tool_call = ToolCall(
                            tool_name=call_data.get('tool_name') or call_data.get('tool', ''),
                            parameters=call_data.get('parameters', {}),
                            intent=intent,
                            rationale=call_data.get('rationale', ''),
                            order=call_data.get('order', i + 1)
                        )
                        tool_calls.append(tool_call)
                    
                    return ToolPlan(
                        calls=tool_calls,
                        explanation=plan_data.get('explanation', ''),
                        estimated_risk=plan_data.get('estimated_risk', 'medium'),
                        requires_confirmation=plan_data.get('requires_confirmation', False)
                    )
        except Exception as e:
            logger.error(f"Tool planning failed: {e}")
        
        # Fallback: create a simple plan based on intent
        return self._create_fallback_plan(user_request, intent)
    
    def _create_fallback_plan(self, user_request: str, intent: ToolIntent) -> ToolPlan:
        """Create a fallback plan when LLM planning fails."""
        
        if intent == ToolIntent.SYSTEM_INFO and "system_info" in self.available_tools:
            return ToolPlan(
                calls=[ToolCall(
                    tool_name="system_info",
                    parameters={"info_type": "overview"},
                    intent=intent,
                    rationale="Gather system information"
                )],
                explanation="Show system information",
                estimated_risk="low"
            )
        
        elif intent == ToolIntent.GIT_OPERATIONS and "git_tool" in self.available_tools:
            # If the request hints at problems, produce a troubleshooting sequence
            t = (user_request or "").lower()
            calls: List[ToolCall] = []
            if any(k in t for k in ["conflict", "broken", "issue", "problem", "fail", "merge"]):
                # status -> log -> optional diff
                calls.append(ToolCall(
                    tool_name="git_tool",
                    parameters={"subcommand": "status"},
                    intent=intent,
                    rationale="Check repository status"
                ))
                calls.append(ToolCall(
                    tool_name="git_tool",
                    parameters={"subcommand": "log", "args": ["-n", "5", "--oneline"]},
                    intent=intent,
                    rationale="Show recent commits",
                    order=2
                ))
                if "conflict" in t or "merge" in t:
                    calls.append(ToolCall(
                        tool_name="git_tool",
                        parameters={"subcommand": "diff"},
                        intent=intent,
                        rationale="Inspect conflicting changes",
                        order=3
                    ))
                return ToolPlan(
                    calls=calls,
                    explanation="Git troubleshooting steps",
                    estimated_risk="low"
                )
            # Default single status
            return ToolPlan(
                calls=[ToolCall(
                    tool_name="git_tool",
                    parameters={"subcommand": "status"},
                    intent=intent,
                    rationale="Check git repository status"
                )],
                explanation="Show git status",
                estimated_risk="low"
            )
        
        elif intent == ToolIntent.CODE_SEARCH and "repo_grep" in self.available_tools:
            # Try to extract search pattern from request
            words = user_request.split()
            pattern = words[-1] if words else "TODO"  # Simple heuristic
            
            return ToolPlan(
                calls=[ToolCall(
                    tool_name="repo_grep",
                    parameters={"pattern": pattern, "path": "."},
                    intent=intent,
                    rationale=f"Search for '{pattern}' in repository"
                )],
                explanation=f"Search repository for '{pattern}'",
                estimated_risk="low"
            )
        
        # If we can infer multiple actions from the request, create a basic multi-step plan
        calls: List[ToolCall] = []
        order = 1
        text = (user_request or "").lower()
        
        # Heuristic: disk usage check
        if any(k in text for k in ["disk usage", "disk space", "df -h", "free space","check disk"]):
            calls.append(ToolCall(
                tool_name="command_executor",
                parameters={"command": "df -h", "explain_only": True},
                intent=ToolIntent.COMMAND_EXECUTION,
                rationale="Check disk usage",
                order=order,
            ))
            order += 1
        
        # Heuristic: list directory
        if any(k in text for k in ["list", "ls", "show files", "list files", "directory listing"]):
            # Try to extract a path (very simple heuristic)
            path = "."
            for token in text.split():
                if token.startswith("/") or token.startswith("./"):
                    path = token
                    break
            calls.append(ToolCall(
                tool_name="file_manager",
                parameters={"operation": "list", "path": path},
                intent=ToolIntent.FILE_OPERATIONS,
                rationale=f"List directory {path}",
                order=order,
            ))
            order += 1
        
        # Heuristic: system overview
        if any(k in text for k in ["system info", "system information", "overview", "cpu", "memory"]):
            calls.append(ToolCall(
                tool_name="system_info",
                parameters={"info_type": "overview"},
                intent=ToolIntent.SYSTEM_INFO,
                rationale="Get system overview",
                order=order,
            ))
            order += 1
        
        # Heuristic: simple code/text search
        if any(k in text for k in ["search", "find pattern", "grep"]):
            # Extract a simple pattern (last word after 'search' or quoted string)
            pattern = None
            import re as _re
            m = _re.search(r'"([^"]+)"', user_request or "")
            if m:
                pattern = m.group(1)
            else:
                m2 = _re.search(r'search\s+for\s+([\w\-\.\_/]+)', text)
                if m2:
                    pattern = m2.group(1)
            if not pattern:
                # fallback to last word
                words = text.split()
                pattern = words[-1] if words else "TODO"
            calls.append(ToolCall(
                tool_name="repo_grep",
                parameters={"pattern": pattern, "path": "."},
                intent=ToolIntent.CODE_SEARCH,
                rationale=f"Search repository for '{pattern}'",
                order=order,
            ))
            order += 1

        # Heuristic: network diagnostics
        if any(k in text for k in ["network", "connectivity", "ping", "latency", "port", "listening", "connection issue", "socket"]):
            # Prefer a safe explain-only diagnostic sequence
            # 1) Check listening ports
            calls.append(ToolCall(
                tool_name="command_executor",
                parameters={"command": "ss -tulpn", "explain_only": True},
                intent=ToolIntent.COMMAND_EXECUTION,
                rationale="Inspect listening ports and sockets",
                order=order,
            ))
            order += 1
            # 2) Ping a reliable target
            calls.append(ToolCall(
                tool_name="command_executor",
                parameters={"command": "ping -c 3 8.8.8.8", "explain_only": True},
                intent=ToolIntent.COMMAND_EXECUTION,
                rationale="Ping external host to test connectivity",
                order=order,
            ))
            order += 1

        # Heuristic: package/dependency overview
        if any(k in text for k in ["python packages", "pip", "dependencies", "packages", "node modules", "npm"]):
            if "npm" in text or "node" in text:
                pkg_cmd = "npm list --depth=0"
            else:
                pkg_cmd = "pip list"
            calls.append(ToolCall(
                tool_name="command_executor",
                parameters={"command": pkg_cmd, "explain_only": True},
                intent=ToolIntent.COMMAND_EXECUTION,
                rationale="List installed packages",
                order=order,
            ))
            order += 1

        # Heuristic: git troubleshooting flow
        if ("git" in text) and any(k in text for k in ["conflict", "broken", "issue", "problem", "fail", "merge"]):
            calls.append(ToolCall(
                tool_name="git_tool",
                parameters={"subcommand": "status"},
                intent=ToolIntent.GIT_OPERATIONS,
                rationale="Check repository status",
                order=order,
            ))
            order += 1
            # Follow with recent commits or diff
            calls.append(ToolCall(
                tool_name="git_tool",
                parameters={"subcommand": "log", "args": ["-n", "5", "--oneline"]},
                intent=ToolIntent.GIT_OPERATIONS,
                rationale="Show recent commits",
                order=order,
            ))
            order += 1
            # If conflicts are mentioned, show diff
            if "conflict" in text or "merge" in text:
                calls.append(ToolCall(
                    tool_name="git_tool",
                    parameters={"subcommand": "diff"},
                    intent=ToolIntent.GIT_OPERATIONS,
                    rationale="Inspect conflicting changes",
                    order=order,
                ))
                order += 1

        # Heuristic: simple recovery step when errors mentioned
        if any(k in text for k in ["error", "failed", "fails", "timeout"]):
            # Add a grep step to search codebase/logs for the error keyword
            key = None
            import re as _re2
            merr = _re2.search(r'error\s*[:\-]?\s*([\w\-\.\_/]+)', text)
            if merr:
                key = merr.group(1)
            if not key:
                key = "error"
            calls.append(ToolCall(
                tool_name="repo_grep",
                parameters={"pattern": key, "path": "."},
                intent=ToolIntent.CODE_SEARCH,
                rationale=f"Search repository for error keyword '{key}'",
                order=order,
            ))
            order += 1
        
        if calls:
            return ToolPlan(
                calls=calls,
                explanation="Multi-step plan based on request heuristics",
                estimated_risk="low",
                requires_confirmation=False,
            )
        
        # Default fallback
        return ToolPlan(
            calls=[],
            explanation="Unable to create execution plan",
            estimated_risk="low",
            requires_confirmation=True
        )
    
    async def execute_plan(self, plan: ToolPlan) -> List[ToolResult]:
        """Execute a tool plan and return results."""
        results = []
        
        # Sort calls by order
        sorted_calls = sorted(plan.calls, key=lambda x: x.order)
        
        for call in sorted_calls:
            if call.tool_name not in self.available_tools:
                result = ToolResult(
                    success=False,
                    content="",
                    error=f"Tool '{call.tool_name}' not available"
                )
                results.append(result)
                continue
            
            tool = self.available_tools[call.tool_name]
            
            try:
                result = await tool.execute(call.parameters)
                results.append(result)
                
                # If a step fails and it's critical, stop execution
                if not result.success and call.rationale.lower().startswith('critical'):
                    logger.warning(f"Critical step failed: {result.error}")
                    break
                    
            except Exception as e:
                error_result = ToolResult(
                    success=False,
                    content="",
                    error=f"Tool execution failed: {str(e)}"
                )
                results.append(error_result)
                logger.error(f"Tool {call.tool_name} failed: {e}")
        
        return results
    
    def register_tool(self, name: str, tool: BaseTool):
        """Register a new tool for selection."""
        self.available_tools[name] = tool
        self.tool_schemas = self._generate_tool_schemas()
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return list(self.available_tools.keys())


# Utility function for easy integration
async def smart_tool_selection(
    user_request: str, 
    llm, 
    tools: Dict[str, BaseTool], 
    context: Optional[Dict[str, Any]] = None
) -> Tuple[ToolPlan, List[ToolResult]]:
    """
    Convenience function for smart tool selection and execution.
    
    Returns:
        Tuple of (plan, results) where plan describes what was planned
        and results contains the execution results.
    """
    selector = SmartToolSelector(llm, tools)
    plan = await selector.create_tool_plan(user_request, context)
    results = await selector.execute_plan(plan)
    
    return plan, results
