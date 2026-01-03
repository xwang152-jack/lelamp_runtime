import asyncio
import logging
import inspect
import typing
from contextlib import AsyncExitStack
from typing import Any, Callable, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from livekit.agents import function_tool

logger = logging.getLogger("lelamp.mcp")

class McpManager:
    def __init__(self):
        self._exit_stack = AsyncExitStack()
        self._sessions: List[ClientSession] = []
        self._tools: Dict[str, Callable] = {}

    async def connect_stdio(self, name: str, command: str, args: list[str], env: dict[str, str] | None = None) -> ClientSession:
        logger.info(f"Connecting to MCP server '{name}': {command} {args}")
        try:
            server_params = StdioServerParameters(command=command, args=args, env=env)
            read, write = await self._exit_stack.enter_async_context(stdio_client(server_params))
            session = await self._exit_stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self._sessions.append(session)
            logger.info(f"Connected to MCP server '{name}'")
            return session
        except Exception as e:
            logger.error(f"Failed to connect to MCP server '{name}': {e}")
            raise

    async def get_bridged_tools(self) -> Dict[str, Callable]:
        """
        Discover tools from all connected MCP sessions and create LiveKit-compatible function tools.
        Returns a dict mapping tool name to the wrapped callable.
        """
        bridged_tools = {}
        
        for session in self._sessions:
            try:
                result = await session.list_tools()
                for tool in result.tools:
                    # Avoid name collisions, maybe prefix?
                    # For now, use raw name
                    tool_name = tool.name
                    if tool_name in bridged_tools:
                        logger.warning(f"Duplicate tool name {tool_name}, skipping")
                        continue
                        
                    # Create the bridge function
                    func = self._create_bridge_function(session, tool)
                    bridged_tools[tool_name] = func
            except Exception as e:
                logger.error(f"Error listing tools from session: {e}")
                
        self._tools.update(bridged_tools)
        return bridged_tools

    def _create_bridge_function(self, session: ClientSession, tool_info: Any) -> Callable:
        # tool_info has name, description, inputSchema
        
        async def dynamic_tool_func(*args, **kwargs):
            # Map args/kwargs back to tool call
            # For simplicity, we assume kwargs are used because we'll define signature with named params
            logger.info(f"Calling MCP tool {tool_info.name} with {kwargs}")
            try:
                result = await session.call_tool(tool_info.name, arguments=kwargs)
                # Result is CallToolResult
                content_list = result.content
                output = []
                for item in content_list:
                    if item.type == "text":
                        output.append(item.text)
                    elif item.type == "image":
                        output.append(f"[Image: {item.mimeType}]") 
                    elif item.type == "resource":
                        output.append(f"[Resource: {item.uri}]")
                return "\n".join(output)
            except Exception as e:
                return f"Error calling MCP tool {tool_info.name}: {str(e)}"

        # Build dynamic signature
        sig = self._build_signature(tool_info.inputSchema)
        dynamic_tool_func.__signature__ = sig
        dynamic_tool_func.__name__ = tool_info.name
        dynamic_tool_func.__doc__ = tool_info.description or f"Call MCP tool {tool_info.name}"
        
        # We also need to set __annotations__ for some inspectors
        dynamic_tool_func.__annotations__ = {
            p.name: p.annotation for p in sig.parameters.values() 
            if p.annotation is not inspect.Parameter.empty
        }
        dynamic_tool_func.__annotations__['return'] = str

        return function_tool(dynamic_tool_func)

    def _build_signature(self, schema: Dict[str, Any]) -> inspect.Signature:
        parameters = []
        props = schema.get("properties", {})
        required = set(schema.get("required", []))
        
        for prop_name, prop_schema in props.items():
            # Map JSON schema types to Python types
            py_type = self._map_json_type(prop_schema.get("type"), prop_schema)
            
            default = inspect.Parameter.empty
            if prop_name not in required:
                default = prop_schema.get("default", None)
                if default is None:
                     default = None
                     if py_type is not inspect.Parameter.empty:
                         py_type = Optional[py_type]

            param = inspect.Parameter(
                name=prop_name,
                kind=inspect.Parameter.KEYWORD_ONLY, 
                default=default,
                annotation=py_type
            )
            parameters.append(param)
            
        return inspect.Signature(parameters=parameters)

    def _map_json_type(self, json_type: str | list, schema: Dict) -> Any:
        if isinstance(json_type, list):
            types = [self._map_json_type(t, schema) for t in json_type if t != "null"]
            if not types:
                return Any
            if len(types) == 1:
                return types[0]
            return typing.Union[tuple(types)]

        if json_type == "string":
            return str
        elif json_type == "integer":
            return int
        elif json_type == "number":
            return float
        elif json_type == "boolean":
            return bool
        elif json_type == "array":
            return list
        elif json_type == "object":
            return dict
        return Any

    async def aclose(self):
        await self._exit_stack.aclose()
