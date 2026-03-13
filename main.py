"""
finz — Streamlit + Groq + FastMCP (optional)
pip install streamlit groq fastmcp nest_asyncio
streamlit run app.py
"""

import json
import asyncio
import nest_asyncio
import streamlit as st
from groq import Groq
from fastmcp import Client

nest_asyncio.apply()

# ── Config ───────────────────────────────────────────────────
st.set_page_config(
   page_title="FinZee",
   page_icon="🧊",
   layout="wide",
)
st.subheader(":green-background[FinZee] > :green[MCP]-Powered Financial :green[AI]gent", divider='green')


# ── Secrets ───────────────────────────────────────────────────
MCP_URL = "https://finzee.fastmcp.app/mcp"
MODEL   = "openai/gpt-oss-20b"

groq_key  = st.secrets.get("GROQ_API_KEY", "")
mcp_token = st.secrets.get("MCP_TOKEN", "")

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:

    # Keys status
    st.markdown("### Keys")
    if groq_key:
        st.success("🟢 GROQ_API_KEY set")
    else:
        st.error("✗ GROQ_API_KEY missing")
    if mcp_token:
        st.success("🟢 MCP_TOKEN set")
    else:
        st.warning("○ MCP_TOKEN not set")

    st.divider()

    # MCP toggle
    st.markdown("### MCP Tools")
    use_mcp = st.toggle("Enable MCP (finz server)", value=False)
    if use_mcp:
        st.caption(f"`{MCP_URL}`")

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.mcp_was_on = None
        st.rerun()

# ── Reset chat when MCP mode changes ─────────────────────────
if "mcp_was_on" not in st.session_state:
    st.session_state.mcp_was_on = use_mcp

if st.session_state.mcp_was_on != use_mcp:
    st.session_state.messages = []
    st.session_state.mcp_was_on = use_mcp
    st.rerun()

# ── Chat state ────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

mode_label = "🛠️ MCP enabled 🟢" if use_mcp else "💬 Standard chat"
st.caption(mode_label)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── MCP agentic loop ──────────────────────────────────────────
async def run_with_mcp(token: str, prompt_messages: list, groq_client: Groq):
    steps = []
    async with Client(MCP_URL, auth=token) as mcp:
        raw_tools = await mcp.list_tools()
        steps.append(f"📋 {len(raw_tools)} tools: {[t.name for t in raw_tools]}")
        tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.inputSchema,
                },
            }
            for t in raw_tools
        ]
        messages = prompt_messages.copy()
        response = groq_client.chat.completions.create(
            model=MODEL, messages=messages, tools=tools, tool_choice="auto"
        )
        resp_msg = response.choices[0].message
        while resp_msg.tool_calls:
            messages.append(resp_msg)
            for tc in resp_msg.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments)
                steps.append(f"🛠️ `{fn_name}` ← `{fn_args}`")
                result = await mcp.call_tool(fn_name, fn_args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, default=str),
                })
            steps.append("💬 Generating answer...")
            response = groq_client.chat.completions.create(
                model=MODEL, messages=messages, tools=tools
            )
            resp_msg = response.choices[0].message
    return resp_msg.content or "No response.", steps

# ── Input ─────────────────────────────────────────────────────
if prompt := st.chat_input("Ask anything — e.g. Latest news on NVDA"):
    if not groq_key:
        st.error("Add your Groq API key in the sidebar.")
        st.stop()
    if use_mcp and not mcp_token:
        st.error("Add your FastMCP token to use MCP tools.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    system = {
        "role": "system",
        "content": (
            "You are an expert financial analyst with access to real-time market data tools. "
            "IMPORTANT RULES:\n"
            "- ALWAYS call a tool before answering any question about a stock, ticker, price, earnings, filings, or market data.\n"
            "- Never answer from memory when a tool can provide fresher data.\n"
            "- If the user mentions a ticker or company, call the appropriate tool immediately.\n"
            "- Call multiple tools in sequence if needed (e.g. get_ticker then get_news).\n"
            "- After tool results, give a concise structured answer highlighting key figures.\n"
            "- If no tool is relevant, answer directly and say why."
        ) if use_mcp else (
            "You are an expert financial analyst assistant. "
            "Answer clearly and concisely, highlighting key figures and metrics. "
            "Be precise with numbers and mention the timeframe when relevant."
        ),
    }
    prompt_messages = [system, *[{"role": m["role"], "content": m["content"]}
                                  for m in st.session_state.messages]]

    with st.chat_message("assistant"):
        answer  = ""
        thinking = None

        if use_mcp:
            with st.status("Thinking...", expanded=True) as status:
                try:
                    groq_client = Groq(api_key=groq_key)
                    loop = asyncio.get_event_loop()
                    answer, steps = loop.run_until_complete(
                        run_with_mcp(mcp_token, prompt_messages, groq_client)
                    )
                    for step in steps:
                        st.write(step)
                    status.update(label="Done ✓", state="complete", expanded=False)
                except Exception as e:
                    answer = f"Error: {e}"
                    status.update(label="Error", state="error")
        else:
            try:
                groq_client = Groq(api_key=groq_key)
                response = groq_client.chat.completions.create(
                    model=MODEL, messages=prompt_messages
                )
                msg_obj  = response.choices[0].message
                answer   = msg_obj.content or "No response."
                thinking = getattr(msg_obj, "reasoning_content", None)
            except Exception as e:
                answer = f"Error: {e}"

        if thinking:
            with st.expander("🧠 Thinking", expanded=False):
                st.markdown(thinking)
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
