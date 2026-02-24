"""
Medical Literature Knowledge Graph — Chainlit Chat UI

Runs on port 8002 (use: chainlit run app.py --host 0.0.0.0 --port 8002).

Environment variables:
  MCP_SSE_URL       URL of the MCP SSE server (default: http://localhost/mcp/sse)
  LLM_PROVIDER      anthropic | openai | ollama  (default: anthropic)
  ANTHROPIC_MODEL   (default: claude-sonnet-4-6)
  ANTHROPIC_API_KEY
  OPENAI_MODEL      (default: gpt-4o)
  OPENAI_API_KEY
  OLLAMA_MODEL      (default: llama3.2)
  OLLAMA_BASE_URL   (default: http://ollama:11434)
  EXAMPLES_FILE     path to YAML file of example prompts (default: examples.yaml)
"""

import asyncio
import json
import os
from contextlib import AsyncExitStack
from typing import Any

import yaml
import chainlit as cl
from chainlit.input_widget import Select
import litellm
from mcp import ClientSession
from mcp.client.sse import sse_client

# ── Configuration ────────────────────────────────────────────────────────────

MCP_SSE_URL   = os.environ.get("MCP_SSE_URL", "http://localhost/mcp/sse")
LLM_PROVIDER  = os.environ.get("LLM_PROVIDER", "anthropic").lower()
EXAMPLES_FILE = os.environ.get("EXAMPLES_FILE", "examples.yaml")

SYSTEM_PROMPT = """You are an expert assistant for a medical literature knowledge graph.
You have access to tools that can query a graph database of medical research papers,
extract entities, find relationships, and surface evidence for clinical questions.
Always cite the papers you draw evidence from when possible."""


def get_litellm_model() -> dict[str, Any]:
    """Return the model string and any extra kwargs for litellm.completion."""
    if LLM_PROVIDER == "anthropic":
        return {
            "model": os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        }
    elif LLM_PROVIDER == "openai":
        return {
            "model": os.environ.get("OPENAI_MODEL", "gpt-4o"),
        }
    elif LLM_PROVIDER == "ollama":
        return {
            "model": f"ollama/{os.environ.get('OLLAMA_MODEL', 'llama3.2')}",
            "api_base": os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434"),
        }
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER!r}")


# ── Examples ─────────────────────────────────────────────────────────────────

def load_examples() -> dict[str, str]:
    """Load examples from YAML: {label: prompt_text}"""
    try:
        with open(EXAMPLES_FILE) as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict):
            return data
        if isinstance(data, list):
            # Support list-of-{label, prompt} dicts too
            return {item["label"]: item["prompt"] for item in data}
    except FileNotFoundError:
        pass
    # Fallback built-in examples
    return {
        "HIV treatment resistance mechanisms": (
            "What does the literature say about mechanisms of HIV resistance to antiretroviral therapy?"
        ),
        "CD4 count as prognostic marker": (
            "Summarize evidence on CD4 T-cell count as a prognostic marker in HIV/AIDS progression."
        ),
        "Drug-drug interactions in ART": (
            "What are the most clinically significant drug-drug interactions in antiretroviral therapy regimens?"
        ),
        "Long COVID neurological symptoms": (
            "What patterns of neurological symptoms appear in long COVID, according to recent literature?"
        ),
        "Graph: Find related conditions": (
            "Use the knowledge graph to find conditions strongly related to HIV-associated neurocognitive disorder."
        ),
    }


EXAMPLES: dict[str, str] = load_examples()
EXAMPLE_PLACEHOLDER = "— select an example —"


# ── MCP session management ────────────────────────────────────────────────────

async def create_mcp_session() -> tuple[ClientSession, AsyncExitStack]:
    stack = AsyncExitStack()
    read, write = await stack.enter_async_context(sse_client(MCP_SSE_URL))
    session = await stack.enter_async_context(ClientSession(read, write))
    await session.initialize()
    return session, stack


def mcp_tools_to_litellm(tools) -> list[dict]:
    """Convert MCP tool descriptors to OpenAI-style tool dicts for litellm."""
    result = []
    for t in tools:
        result.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.inputSchema if t.inputSchema else {"type": "object", "properties": {}},
            },
        })
    return result


# ── Chainlit lifecycle ────────────────────────────────────────────────────────

@cl.on_chat_start
async def on_chat_start():
    # Connect to MCP server
    try:
        session, stack = await create_mcp_session()
        tools_result = await session.list_tools()
        tools = tools_result.tools
        cl.user_session.set("mcp_session", session)
        cl.user_session.set("mcp_stack", stack)
        cl.user_session.set("mcp_tools", tools)
        cl.user_session.set("litellm_tools", mcp_tools_to_litellm(tools))
        tool_names = [t.name for t in tools]
        status = f"✅ Connected to MCP server — {len(tools)} tools: `{'`, `'.join(tool_names)}`"
    except Exception as e:
        cl.user_session.set("mcp_session", None)
        cl.user_session.set("mcp_tools", [])
        cl.user_session.set("litellm_tools", [])
        status = f"⚠️ Could not connect to MCP server ({MCP_SSE_URL}): {e}"

    # Initialize message history
    cl.user_session.set("history", [{"role": "system", "content": SYSTEM_PROMPT}])

    # Build the examples dropdown as a Chat Setting
    example_labels = [EXAMPLE_PLACEHOLDER] + list(EXAMPLES.keys())
    await cl.ChatSettings(
        [
            Select(
                id="example",
                label="📋 Example prompts",
                values=example_labels,
                initial_value=EXAMPLE_PLACEHOLDER,
            )
        ]
    ).send()

    provider_label = LLM_PROVIDER.upper()
    model_info = get_litellm_model()["model"]
    await cl.Message(
        content=f"{status}\n\n**LLM:** `{provider_label}` → `{model_info}`\n\n"
                "Ask me anything about the medical literature knowledge graph, "
                "or pick an example from the dropdown above ↑"
    ).send()


@cl.on_settings_update
async def on_settings_update(settings: dict):
    selected_label = settings.get("example", EXAMPLE_PLACEHOLDER)
    if selected_label == EXAMPLE_PLACEHOLDER:
        return

    prompt_text = EXAMPLES.get(selected_label)
    if not prompt_text:
        return

    # Show the chosen example as a user message and run it
    await cl.Message(content=prompt_text, author="User").send()
    await run_chat(prompt_text)


@cl.on_message
async def on_message(message: cl.Message):
    await run_chat(message.content)


@cl.on_chat_end
async def on_chat_end():
    stack: AsyncExitStack | None = cl.user_session.get("mcp_stack")
    if stack:
        await stack.aclose()


# ── Core chat loop ────────────────────────────────────────────────────────────

async def run_chat(user_text: str):
    history: list[dict] = cl.user_session.get("history")
    history.append({"role": "user", "content": user_text})

    mcp_session: ClientSession | None = cl.user_session.get("mcp_session")
    litellm_tools: list[dict] = cl.user_session.get("litellm_tools", [])
    llm_kwargs = get_litellm_model()

    thinking_msg = cl.Message(content="")
    await thinking_msg.send()

    # Agentic tool-call loop
    max_iterations = 8
    for _ in range(max_iterations):
        response = await asyncio.to_thread(
            litellm.completion,
            messages=history,
            tools=litellm_tools if litellm_tools else None,
            tool_choice="auto" if litellm_tools else None,
            stream=False,
            **llm_kwargs,
        )

        msg = response.choices[0].message
        history.append(msg.model_dump(exclude_none=True))

        # If the LLM wants to call tools
        if msg.tool_calls:
            tool_results = await execute_tool_calls(msg.tool_calls, mcp_session, thinking_msg)
            history.extend(tool_results)
            continue  # loop back with tool results

        # Final text response
        final_text = msg.content or ""
        await thinking_msg.update()
        thinking_msg.content = final_text
        await thinking_msg.update()
        break

    cl.user_session.set("history", history)


async def execute_tool_calls(
    tool_calls, mcp_session: ClientSession | None, status_msg: cl.Message
) -> list[dict]:
    """Run each tool call against the MCP server, return tool-result messages."""
    result_messages = []

    for tc in tool_calls:
        fn_name = tc.function.name
        try:
            fn_args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            fn_args = {}

        # Show a spinner/status for the tool call
        async with cl.Step(name=f"🔧 {fn_name}", type="tool") as step:
            step.input = json.dumps(fn_args, indent=2)

            if mcp_session is None:
                tool_output = {"error": "MCP session not available"}
            else:
                try:
                    mcp_result = await mcp_session.call_tool(fn_name, fn_args)
                    # MCP result content is a list of content blocks
                    tool_output = " ".join(
                        block.text for block in mcp_result.content
                        if hasattr(block, "text")
                    ) or str(mcp_result.content)
                except Exception as e:
                    tool_output = f"Tool error: {e}"

            step.output = tool_output if isinstance(tool_output, str) else json.dumps(tool_output)

        result_messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "name": fn_name,
            "content": tool_output if isinstance(tool_output, str) else json.dumps(tool_output),
        })

    return result_messages
