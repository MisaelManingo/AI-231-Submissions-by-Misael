#!/usr/bin/env python3
"""
MCP-to-Pipecat Function Calling Bridge
Extracts tool schemas from MCPServer (MCP 2.x) and registers them with
Pipecat's GeminiLiveLLMService, complete with turn latency instrumentation.
"""

import time
import inspect
import asyncio
from pipecat.adapters.llm.schemas.function_schema import FunctionSchema
from pipecat.services.llm_service import FunctionCallParams


def get_mcp_function_schemas(mcp_server) -> list[FunctionSchema]:
    """Extracts all registered tools from MCPServer and converts to Pipecat FunctionSchemas."""
    schemas = []
    tools = mcp_server._tool_manager.list_tools()

    for t in tools:
        raw_params = getattr(t, "parameters", {}) or {}
        properties = raw_params.get("properties", {})
        required = raw_params.get("required", [])

        # Clean JSON Schema properties for Gemini
        clean_props = {}
        for prop_name, prop_val in properties.items():
            clean_props[prop_name] = {
                "type": prop_val.get("type", "string"),
                "description": prop_val.get("description", prop_val.get("title", prop_name))
            }
            if "default" in prop_val:
                clean_props[prop_name]["default"] = prop_val["default"]

        schema = FunctionSchema(
            name=t.name,
            description=t.description or f"Tool {t.name}",
            properties=clean_props,
            required=required
        )
        schemas.append(schema)

    return schemas


def bridge_mcp_to_pipecat(mcp_server, llm_service, benchmark_logger=None, hardware_manager=None):
    """
    Registers each tool in mcp_server to llm_service with benchmark timing
    and LCD status reporting.
    """
    tools = mcp_server._tool_manager.list_tools()
    tool_map = {t.name: t for t in tools}

    print(f"\n[MCP BRIDGE]: Bridging {len(tools)} MCP tools to GeminiLiveLLMService:")
    for t in tools:
        tool_name = t.name
        tool_obj = t
        print(f" - Registered: {tool_name}")

        def make_handler(name, tool):
            async def handler(params: FunctionCallParams):
                t_start = time.perf_counter()
                print(f"⚡ [MCP TOOL CALL]: Executing '{name}' with args {params.arguments}", flush=True)

                if hardware_manager:
                    hardware_manager.on_thinking(f"Tool: {name[:12]}")

                if benchmark_logger:
                    benchmark_logger.record_tool_start(name, t_start)

                result_val = None
                try:
                    # Run tool function (handles sync or async)
                    kwargs = dict(params.arguments or {})
                    if inspect.iscoroutinefunction(tool.fn):
                        result_val = await tool.fn(**kwargs)
                    else:
                        result_val = await asyncio.to_thread(tool.fn, **kwargs)
                except Exception as e:
                    print(f"❌ [MCP TOOL ERROR] {name}: {e}", flush=True)
                    result_val = f"Error executing tool {name}: {e}"
                finally:
                    t_finish = time.perf_counter()
                    duration_ms = (t_finish - t_start) * 1000.0
                    print(f"✅ [MCP TOOL FINISHED]: '{name}' ({duration_ms:.1f}ms) -> {str(result_val)[:100]}", flush=True)

                    if benchmark_logger:
                        benchmark_logger.record_tool_finish(name, t_finish)

                # Return result to Gemini Live via result_callback
                await params.result_callback(str(result_val))
                print(f"📡 [MCP RESULT DELIVERED]: '{name}' result delivered to Gemini Live.", flush=True)

            return handler

        llm_service.register_function(tool_name, make_handler(tool_name, tool_obj))

    print("[MCP BRIDGE]: All MCP tools successfully wired to Gemini Live function calling.\n")

