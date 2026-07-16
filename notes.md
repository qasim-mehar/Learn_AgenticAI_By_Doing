# 🧠 My Agentic AI Learning Notes
> *Written as if your teacher is explaining everything to you — clear, structured, and honest about what you still need to learn.*

---

## 📅 Journey So Far

You started this repo to learn **Agentic AI** by actually building things. Not just reading — building. That's the right approach. Every project here taught you something new. Let's go through everything, one layer at a time.

---

## 🤖 Chapter 1: Gen AI vs Agentic AI

### What is Generative AI?
Generative AI (Gen AI) is a model that **generates content** — text, images, audio — based on your prompt. You give it input, it gives you output. Simple call and response. Think of ChatGPT: you type, it answers. Done.

**The limitation?** It's stateless and passive. It does what you ask, nothing more. It cannot plan, cannot use tools on its own, and cannot make decisions across multiple steps.

### What is Agentic AI?
Agentic AI is an AI system that can **reason, plan, make decisions, and take actions** — step by step — to achieve a goal. It doesn't just answer once; it works through a problem like a real agent.

Key differences:

| Feature | Gen AI | Agentic AI |
|---|---|---|
| Takes one step | ✅ | ✅ |
| Plans multiple steps | ❌ | ✅ |
| Uses tools (search, code, APIs) | ❌ | ✅ |
| Makes decisions based on output | ❌ | ✅ |
| Has memory across turns | ❌ | ✅ |
| Can loop or branch logic | ❌ | ✅ |

> **Think of it this way:** Gen AI is a smart answering machine. Agentic AI is a smart employee who can figure out what to do next, use tools, and finish the job.

---

## 🔗 Chapter 2: What is LangGraph?

**LangGraph** is a Python framework built on top of LangChain that lets you build **stateful, multi-step AI applications** using a concept called a **graph**.

### The Core Idea: Graphs
A graph in computer science is just a collection of **nodes** (steps/functions) connected by **edges** (the flow of data). LangGraph lets you define your AI workflow as a graph.

```
START → Node A → Node B → Node C → END
```

Every node receives the current **state**, does some work (usually calls an LLM), and returns an updated state. The graph moves data from node to node until it reaches `END`.

### Why do we need it?
Because LangChain alone is great for single chains (one LLM call). But real-world AI apps need:
- Multiple LLM calls in sequence
- Parallel processing
- Decision-making (routing to different paths)
- Memory between turns
- Tool usage loops

LangGraph handles all of this cleanly.

---

## ❓ Chapter 3: Why LangGraph, Not Just LangChain?

This is a great question you noted. Here's the honest answer:

**LangChain** is a toolkit — it gives you models, tools, parsers, and chains. But it handles flow linearly. When you need to branch, loop, or maintain complex state, it becomes messy.

**LangGraph** adds a proper **graph execution engine** on top of LangChain. It gives you:

- ✅ **Explicit state management** (you define a schema for your data)
- ✅ **Visual, understandable workflows** (you can even export a PNG of your graph!)
- ✅ **Conditional routing** (decide at runtime which node to go to next)
- ✅ **Parallel execution** (run multiple nodes at the same time)
- ✅ **Memory / checkpointing** (save conversation state between calls)
- ✅ **Built-in tool execution** (`ToolNode`)

**The rule of thumb:** If your AI app has more than one step or needs any kind of decision-making — use LangGraph.

---

## 🏗️ Chapter 4: Workflows and Their Types

A **workflow** is just the pattern in which your nodes are connected and executed. You have learned these types:

### 4.1 Sequential Workflow
**Every node runs one after another.** The output of node A is the input to node B.

```
START → Node A → Node B → Node C → END
```

**Real example — Your `script_writter` project:**

```
START → TextEditor → ScriptWriter → Translator → END
```

1. **TextEditor** cleans up your raw, messy input
2. **ScriptWriter** writes a full YouTube/podcast script
3. **Translator** converts the script to Roman Urdu

Each node depends on the previous one. They cannot run at the same time. This is the simplest and most readable workflow pattern.

**Code pattern:**
```python
graph.add_edge(START, "TextEditor")
graph.add_edge("TextEditor", "ScriptWritter")
graph.add_edge("ScriptWritter", "Translator")
graph.add_edge("Translator", END)
```

---

### 4.2 Parallel Workflow (Fan-Out / Fan-In)
**Multiple nodes run at the SAME TIME.** The graph "fans out" from one point to many nodes in parallel, then "fans in" to collect results back together.

```
         → Node A →
START →  → Node B →  → END
         → Node C →
```

**Real example — Your `AI_content_Moderator` project:**

```
         → toxicity_node    →
START →  → cultural_node    →  → END
         → copyright_node   →
```

All three checker nodes run **simultaneously** on the same input text. This is efficient — instead of running one after another (which would take 3x longer), they all run in parallel and the results are collected at the end.

**Code pattern:**
```python
# Fan-Out (from START to multiple nodes)
graph_builder.add_edge(START, "toxicity_node")
graph_builder.add_edge(START, "cultural_node")
graph_builder.add_edge(START, "copyright_node")

# Fan-In (from multiple nodes to END)
graph_builder.add_edge("toxicity_node", END)
graph_builder.add_edge("cultural_node", END)
graph_builder.add_edge("copyright_node", END)
```

---

### 4.3 Conditional Workflow
**The graph decides which node to go to next based on the output of a node.** This is where AI gets truly "agentic" — it can route itself.

```
START → LLM Node → [router function] → Node A (if condition X)
                                     → Node B (if condition Y)
                                     → END    (if done)
```

**Real example — Your `chat_bot` project (tools_condition):**

```
START → chatbotWithTools → [tools_condition] → tools node (if tool call needed)
                                             → END        (if answer is ready)
```

The `tools_condition` function checks: did the LLM want to call a tool? If yes → go to `tools` node. If no → go to `END`. After the tools node runs, it loops back to the chatbot. This is also called an **agent loop**.

**Code pattern:**
```python
graph_builder_with_tools.add_conditional_edges("chatbotWithTools", tools_condition)
graph_builder_with_tools.add_edge("tools", "chatbotWithTools")
```

---

## 📦 Chapter 5: State in LangGraph

**State** is the shared data container that flows through your entire graph. Every node reads from it and writes back to it. Think of it as the "memory" of your current run.

You define state using Python's `TypedDict`:

```python
from typing import TypedDict

class MyState(TypedDict):
    user_input: str
    result: str
```

Each node receives the full state and returns a dictionary of **only the fields it changed**:

```python
def my_node(state: MyState) -> dict:
    # Read from state
    text = state["user_input"]
    # Do work...
    # Return only what changed
    return {"result": "some answer"}
```

LangGraph **merges** this return value back into the state automatically.

---

## 🔄 Chapter 6: Reducers

This is one of the most important concepts you wrote in your README, and it's subtle.

### The Problem Without Reducers
By default, when a node returns a value, LangGraph **replaces** the existing value in state. That works for simple strings. But what about lists (like a conversation history) or dictionaries?

If Node A adds `{"score": {"toxicity": 80}}` and Node B adds `{"score": {"culture": 40}}`, the default behavior would let Node B's update **overwrite** Node A's result. You'd lose the toxicity score!

### The Solution: Reducers
A **Reducer** is a function that defines **how to combine/merge** a new update into the existing value, instead of replacing it.

**Your AI_content_Moderator uses a custom reducer:**
```python
def merge_score_dicts(existing: dict, new_update: dict) -> dict:
    if existing is None:
        return new_update
    return {**existing, **new_update}  # Merge both dicts together
```

And you attach it to the state field using `Annotated`:
```python
from typing import Annotated

class AnalyzerState(TypedDict):
    raw_text: str
    safety_score: Annotated[dict[str, int], merge_score_dicts]
```

Now when three parallel nodes update `safety_score`, instead of overwriting each other, all three scores get **merged** into one dictionary:
```json
{
    "toxicity": 85,
    "cultural_insensitivity": 20,
    "copyright_risk": 10
}
```

**The most common built-in reducer you used:**
```python
from langgraph.graph import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]
```

`add_messages` is LangGraph's built-in reducer for chat history. Instead of replacing the messages list, it **appends** new messages. This is how your chatbot remembers the conversation.

---

## 🌐 Chapter 7: The Router Function and Conditional Edges

In a conditional workflow, you need something to make the decision. That's the **router function**.

### Router Function
A router function receives the current state and returns a **string** — the name of the next node to go to (or `END`).

```python
def my_router(state: State) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"  # go to tools node
    return END          # stop here
```

### Conditional Edges
You wire the router function into the graph using `add_conditional_edges`:

```python
graph.add_conditional_edges(
    "my_llm_node",   # From this node...
    my_router,       # ...call this function to decide...
    {
        "tools": "tools_node",  # if it returns "tools" → go to tools_node
        END: END                # if it returns END → finish
    }
)
```

### `tools_condition` — The built-in router
LangGraph provides `tools_condition` as a ready-made router function. It automatically checks if the LLM's last message contains a **tool call**. If yes, it routes to `"tools"`. If no, it routes to `END`. You used this in your chatbot:

```python
from langgraph.prebuilt import ToolNode, tools_condition

graph_builder.add_conditional_edges("chatbotWithTools", tools_condition)
```

---

## 🔧 Chapter 8: ToolNode

**`ToolNode`** is a pre-built LangGraph node that **executes tools** for you. Instead of writing a node from scratch that calls your functions, you just hand it a list of tools and it handles everything.

```python
from langgraph.prebuilt import ToolNode

tools = [web_search]  # your tool functions
tool_node = ToolNode(tools)

graph.add_node("tools", tool_node)
```

When the LLM says "I want to call `web_search` with argument `topic='AI'`", the `ToolNode` intercepts that tool call, actually calls your `web_search` function, and puts the result back into the messages state as a `ToolMessage`.

### How `@tool` works
You define tools using the `@tool` decorator from LangChain:

```python
from langchain.tools import tool

@tool
def web_search(topic: str) -> str:
    """Search on web for latest information"""
    return tavily_search.invoke(topic)
```

The **docstring is critical** — the LLM reads it to understand when and how to use this tool.

### Binding Tools to an LLM
Before an LLM can use tools, you have to tell it about them:

```python
llm_with_tools = llm.bind_tools(tools)
```

Now the LLM knows what tools are available and can decide to call them.

---

## 💾 Chapter 9: Memory and Checkpointing (MemorySaver)

By default, each time you call `graph.invoke()`, the graph starts fresh — it remembers nothing from previous calls. This is fine for single-use pipelines (like your script writer), but not for a chatbot!

### MemorySaver
`MemorySaver` is LangGraph's in-memory checkpointer. It saves the full state of the graph after every step, keyed by a **thread ID**. This gives you conversation memory.

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)
```

### Thread ID
A `thread_id` is like a session ID. Every unique conversation gets its own thread. When you invoke the graph with the same thread ID, it loads the previous state and continues from where it left off.

```python
config = {"configurable": {"thread_id": "1"}}

# First message
graph.invoke({"messages": [{"role": "user", "content": "my name is qasim"}]}, config=config)

# Second message — the graph REMEMBERS the first message because same thread_id
graph.invoke({"messages": [{"role": "user", "content": "what is my name?"}]}, config=config)
```

Your chatbot demonstrated this perfectly — it remembered "my name is qasim" in the second call because of `MemorySaver` + `thread_id`.

---

## 📡 Chapter 10: Streaming

Instead of waiting for the full graph to finish and then printing the result, you can **stream** the output — receiving updates as each node completes.

```python
for chunk in graph.stream(
    {"messages": [{"role": "user", "content": "hello"}]},
    config=config,
    stream_mode="values",  # Stream the full state after each node
):
    print(chunk)
```

`stream_mode="values"` means: after every node runs, send me the complete current state. This is great for chatbots where you want to show the user a live response.

---

## 🔬 Chapter 11: Project Deep Dives

### Project 1: `script_writter` — Sequential Pipeline

**What it does:** Takes a rough idea in plain English and transforms it into a polished Roman Urdu YouTube script.

**Workflow Type:** Sequential (Linear)

**State:**
```python
class pipleineState(TypedDict):
    raw_input: str       # What you gave it
    edited_input: str    # After grammar/spelling fix
    script_text: str     # After scriptwriting
    roman_urdu: str      # Final output
```

**Nodes:**
1. `text_editor_node` — Fixes grammar, spelling, phrasing
2. `script_writter_node` — Writes a full YouTube script with HOOK, BODY, CTA, and [VISUAL DIRECTIONS]
3. `roman_urdu_node` — Translates narration to Roman Urdu (keeps visual directions in English)

**What you learned here:**
- How to define and use `TypedDict` state
- How nodes pass data forward by returning partial dicts
- How to chain nodes sequentially using `add_edge`
- How to give an LLM a detailed system prompt to control its output format

---

### Project 2: `chat_bot` — Conditional Agent with Tools and Memory

**What it does:** A chatbot that can search the web and remembers conversation history.

**Workflow Type:** Conditional (with a loop) — an agent loop

**State:**
```python
class State(TypedDict):
    messages: Annotated[list, add_messages]  # Full conversation history
```

**Key concepts demonstrated:**
- `add_messages` reducer to accumulate chat history
- `bind_tools` to give the LLM access to tools
- `ToolNode` to execute tool calls automatically
- `tools_condition` for conditional routing
- `MemorySaver` for cross-call memory
- `thread_id` for session management
- `graph.stream()` for streaming output

**The agent loop:**
```
chatbotWithTools
    ↓ (if tool call needed)
tools node
    ↓ (always)
chatbotWithTools
    ↓ (if answer ready)
END
```

---

### Project 3: `AI_content_Moderator` — Parallel Fan-Out/Fan-In

**What it does:** Analyzes a piece of text simultaneously for toxicity, cultural insensitivity, and copyright risk, giving each a score from 0–100.

**Workflow Type:** Parallel (Fan-Out / Fan-In)

**State:**
```python
class AnalyzerState(TypedDict):
    raw_text: str
    safety_score: Annotated[dict[str, int], merge_score_dicts]  # Custom reducer!
```

**Nodes (all run in parallel):**
1. `toxicity_checker_node` → scores hate speech, harassment
2. `cultural_sensitivity_node` → scores cultural bias, stereotypes
3. `copyright_checker_node` → scores IP/plagiarism risk

**Custom Reducer** (`merge_score_dicts`) is the star here — it allows all three parallel nodes to write to the same `safety_score` dict without overwriting each other.

**Final output:**
```json
{
    "toxicity": 85,
    "cultural_insensitivity": 20,
    "copyright_risk": 10
}
```

---

## 🛠️ Chapter 12: Tools and Libraries You Used

| Library | Purpose |
|---|---|
| `langgraph` | Graph-based agentic workflow engine |
| `langchain` | LLM toolkit (tools, prompts, parsers) |
| `langchain_mistralai` | Mistral AI LLM integration |
| `langchain_tavily` | Tavily web search tool |
| `python-dotenv` | Load API keys from `.env` files |
| `rich` | Beautiful terminal output (colored print) |
| `typing.TypedDict` | Define structured state schemas |
| `typing.Annotated` | Attach reducer metadata to state fields |

---

## 🗂️ Chapter 13: Project Setup Patterns

Every project follows the same setup pattern — learn it once:

### 1. Virtual Environment (`.venv`)
Each project has its own isolated Python environment so dependencies don't conflict.

```bash
# Create environment
python -m venv .venv

# Activate (Windows PowerShell)
.venv\Scripts\Activate.ps1
```

### 2. `pyproject.toml` / `requirements.txt`
Lists the packages your project needs.

```bash
pip install -r requirements.txt
# OR
uv sync  # if using pyproject.toml with uv
```

### 3. `.env` file
Stores your secret API keys — **never commit this to Git!**

```
MISTRAL_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
```

### 4. `.gitignore`
Tells Git to ignore `.env`, `.venv`, `__pycache__`, etc. so secrets and bulky folders don't get pushed to GitHub.

---

## 📌 Chapter 14: Things You Are Missing / Still Need to Learn

Here is an honest list of what's not yet in your projects but is essential for real-world Agentic AI:

### Not Covered Yet (Important)

| Topic | What it is |
|---|---|
| **LangSmith** | Tracing and debugging your LangGraph runs — see exactly what every node did, what prompt was sent, what the LLM returned. Essential for debugging complex agents. Your `pyproject.toml` lists it but you haven't used it yet. |
| **Persistent Storage** | `MemorySaver` stores memory in RAM — it's lost when the program stops. For real apps, you need database-backed checkpointers (e.g., `SqliteSaver`, `PostgresSaver`). |
| **Subgraphs** | Embedding one LangGraph inside another. Used for large, modular systems where different parts of an agent are isolated workflows. |
| **Multi-Agent Systems** | Multiple specialized agents (a researcher, a writer, a critic) collaborating — each is its own graph, and a supervisor orchestrates them. |
| **Streaming Tokens** | Currently you stream by node (`stream_mode="values"`). You can also stream individual tokens as the LLM types them, for a ChatGPT-like experience. |
| **Error Handling and Retries** | What happens when an LLM returns something unexpected or an API call fails? Adding retry logic and fallback behavior to nodes. |
| **Custom Tools with APIs** | You used `TavilySearch`. Real agents connect to your own APIs, databases, code interpreters, file systems, etc. |
| **Structured Output** | Instead of parsing `res.content.strip()` manually, use `llm.with_structured_output(MyPydanticModel)` to get typed, validated outputs from LLMs. |

### Skipped for Now (Come Back Later)

| Topic | Why Skipped |
|---|---|
| **Iterative Workflows** | Workflows that loop until a condition is met (e.g., keep refining output until quality is good enough). |
| **Human-in-the-Loop** | Pausing graph execution to ask a human for approval before proceeding. |

---

## 📐 Chapter 15: The Big Picture — How It All Fits Together

```
Your Idea (raw input)
       ↓
   LangGraph Graph
       ↓
   StateGraph(YourState)    ← State holds all your data
       ↓
   Nodes (functions)        ← Each node = one job, one LLM call
       ↓
   Edges                    ← Sequential: one after another
                               Parallel:   fan-out / fan-in
                               Conditional: router function decides
       ↓
   Reducers                 ← Smart merging of updates from parallel nodes
       ↓
   Checkpointer             ← Memory across multiple calls (MemorySaver)
       ↓
   Final State = Your Result
```

Every project you built is a variation of this exact pattern. The more complex your use case, the more of these pieces you combine.

---

# 🧠 My Agentic AI Learning Notes — Extended Chapters

> Continuation of your original notes. Same teaching style, filling the gaps you flagged: streaming modes, HITL/interrupt/Command, MCP (adapters, transports, and what changed in 2026), and the complete RAG pipeline.

---

## 🔄 Chapter 16: Streaming — `stream`, `astream`, and `stream_mode`

You already used `graph.stream(..., stream_mode="values")` in your chatbot. Let's actually understand the full picture, because this is one of those things where "it works" and "I understand why it works" are very different levels.

### `stream` vs `astream` — sync vs async

- `graph.stream(...)` — synchronous. Blocks your program while running. Fine for scripts, notebooks, simple CLIs.
- `graph.astream(...)` — asynchronous. You `await` it inside an `async def`. This is what you need for **FastAPI backends** (like your OrchestraAI backend) because you don't want one user's long agent run to block every other request.

```python
# Sync — blocks the whole process
for chunk in graph.stream({"messages": [...]}, config=config):
    print(chunk)

# Async — used inside FastAPI endpoints
async def run_agent(user_input: str, config: dict):
    async for chunk in graph.astream({"messages": [{"role": "user", "content": user_input}]}, config=config):
        yield chunk  # can be sent to client as it arrives
```

Rule of thumb: **notebook/script → `stream`. Real backend serving multiple users → `astream`.**

### `stream_mode` — what gets sent to you after each step

This is the part most people gloss over. `stream_mode` controls the *shape* of what you receive per chunk, not just whether it streams.

| Mode | What you get | When to use |
|---|---|---|
| `"values"` | The **full current state** after each node | Simple chatbots — you always want the whole picture |
| `"updates"` | Only the **dict returned by the node that just ran** (`{node_name: {...changes...}}`) | Debugging, or when state is large and you only care about deltas |
| `"messages"` | **Token-by-token** LLM output as it's generated, plus metadata about which node/model produced it | ChatGPT-style live typing effect in a UI |
| `"debug"` | Verbose internal events (task start, task end, checkpoints) | Deep debugging of graph execution |
| `"custom"` | Whatever you explicitly push using `get_stream_writer()` inside a node | Progress updates from inside long-running tool calls |

```python
# "values" — full state snapshot every time
for chunk in graph.stream(inputs, config, stream_mode="values"):
    print(chunk["messages"][-1])  # last message in the whole convo so far

# "updates" — only what changed
for chunk in graph.stream(inputs, config, stream_mode="updates"):
    print(chunk)  # e.g. {"chatbotWithTools": {"messages": [...]}}

# "messages" — token streaming (for a live UI)
for msg_chunk, metadata in graph.stream(inputs, config, stream_mode="messages"):
    print(msg_chunk.content, end="", flush=True)

# You can even combine modes:
for mode, chunk in graph.stream(inputs, config, stream_mode=["updates", "messages"]):
    ...
```

**Your mental model:** `"values"` = "give me the whole notebook page," `"updates"` = "give me just the new sentence someone wrote," `"messages"` = "give me the pen strokes as they happen."

---

## ⏸️ Chapter 17: Human-in-the-Loop (HITL), `interrupt`, and `Command`

This is the piece your notes flagged as "skipped for now" — but it's actually central to production agents (loan approvals, sending emails, deleting data — anywhere you don't want the LLM acting fully autonomously).

### Why HITL matters
An agent that can call tools autonomously is powerful but risky. You often want: *"pause right before this specific action, show a human what you're about to do, and only continue if they approve (or let them edit the input first)."*

### `interrupt()` — pausing a node mid-execution

`interrupt()` is a function you call **inside a node**. It pauses the graph at that exact point, and whatever value you pass to it gets surfaced to whoever is running the graph (e.g. your frontend).

```python
from langgraph.types import interrupt, Command

def human_approval_node(state: State) -> dict:
    # Pause here and surface the proposed action to the human
    decision = interrupt({
        "question": "Send this email?",
        "draft": state["draft_email"]
    })
    # Execution PAUSES on the line above until the graph is resumed
    # `decision` will contain whatever the human sends back
    if decision == "approve":
        return {"status": "approved"}
    return {"status": "rejected"}
```

Critically: this **requires a checkpointer** (like `MemorySaver`, which you already know), because pausing means LangGraph has to save the exact state and resume later — potentially minutes or days after.

### Resuming with `Command`

Once paused, you resume the graph by invoking it again with a `Command(resume=...)` object instead of normal input:

```python
config = {"configurable": {"thread_id": "1"}}

# First call — this will run until it hits interrupt() and pause
result = graph.invoke({"messages": [...]}, config=config)
print(result["__interrupt__"])  # shows the payload you passed to interrupt()

# Human reviews, then you resume:
result = graph.invoke(Command(resume="approve"), config=config)
```

`Command` isn't only for resuming interrupts, though — that's its most common use in HITL. More generally, `Command` lets a **node** return both a state update *and* an explicit routing instruction in one object, which is useful when a node needs to decide "update state AND jump to this specific node" without going through `add_conditional_edges`:

```python
def router_node(state: State) -> Command:
    if state["needs_review"]:
        return Command(update={"status": "pending"}, goto="human_review")
    return Command(update={"status": "auto_approved"}, goto="execute")
```

### Real pattern: approve-before-send agent

```
START → draft_email_node → human_approval_node (interrupt) → [approved?] → send_email_node → END
                                                              → [rejected] → END
```

This is the pattern you'll want for OrchestraAI if any agent action has real-world side effects (sending something, deleting something, spending money) — you pause right before the irreversible step.

---

## 🔌 Chapter 18: MCP — Adapters, Transports, and What Changed in 2026

You've got the conceptual foundation already (Host/Client/Server, Tools/Resources/Prompts). Let's get concrete on transports and adapters, and then cover what's actually changing in the protocol right now — because MCP is mid-overhaul as of this writing (July 2026), and a lot of blog posts you'll find are already stale.

### The transports — only two officially, one deprecated

| Transport | How it works | Use case |
|---|---|---|
| **STDIO** | Client spawns the server as a **local subprocess**, talks over stdin/stdout | Local tools: filesystem access, a local Python script, your own DB on your machine |
| **Streamable HTTP** | Client sends HTTP POST requests to a server URL; server can respond with a single JSON response or open a stream for multiple messages | Remote/hosted servers — a SaaS wraps their API as an MCP server you hit over the network |
| **HTTP+SSE** *(deprecated)* | The original remote transport — separate SSE stream for server→client, HTTP POST for client→server | Legacy only. Streamable HTTP replaced it because SSE required a persistent connection and complicated infrastructure. Don't build new servers on this. |

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient({
    # STDIO — local subprocess
    "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/qasim/projects"],
        "transport": "stdio",
    },
    # Streamable HTTP — remote server
    "github": {
        "url": "https://api.githubcopilot.com/mcp/",
        "transport": "streamable_http",
        "headers": {"Authorization": "Bearer <token>"},
    },
})

tools = await client.get_tools()
```

### `langchain-mcp-adapters` — the bridge you already know about

Its whole job: turn MCP tools into `BaseTool` objects LangChain/LangGraph already understands, so `create_react_agent`, `create_agent`, or a custom `ToolNode` can use them exactly like `@tool`-decorated functions. Same idea, no reimplementation on your side.

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(model, tools)  # `tools` came straight from MCP servers
result = await agent.ainvoke({"messages": [{"role": "user", "content": "list files in my repo"}]})
```

### What's actually changing in MCP right now (2026)

The current **stable** spec is still `2025-11-25` — that's what production servers should target today. But a big release candidate (`2026-07-28`, ratifying at the end of this month) is close to final, and it's the largest revision since MCP launched. Worth knowing about now so you're not caught off guard:

- **The protocol is going stateless.** Previously, a Streamable HTTP client did an `initialize` handshake and got back an `Mcp-Session-Id` that every later request had to carry — meaning servers needed sticky sessions and shared session stores to scale. The new RC removes the handshake and session ID entirely, so any request can hit any server instance behind a plain load balancer.
- **MCP Apps** — servers can now ship interactive HTML UIs that render in a sandboxed iframe inside the host app, not just plain text/JSON tool results. (This is the mechanism behind the `[third_party_mcp_app]`-style tools you may have seen mentioned in agent platforms.)
- **Tasks** (for long-running work) moved from a core feature into an official **extension** — `tasks/get`, `tasks/update`, `tasks/cancel` instead of one blocking call.
- **Roots, Sampling, and Logging are deprecated** (not removed yet — minimum 12-month deprecation window) in favor of tool parameters, resource URIs, direct provider APIs, and standard telemetry.
- **Authorization is being hardened** to align more closely with mainstream OAuth 2.1 / OpenID Connect, including an enterprise-managed authorization extension for SSO-based access to many servers at once.

Practical takeaway for you: **build against the stable 2025-11-25 spec today** — that's what `langchain-mcp-adapters` and most servers target. Just don't be surprised in the next few months when you see servers advertising the new stateless behavior; it's not a bug, it's the planned migration.

---

## 📚 Chapter 19: RAG — The Complete Pipeline

Your notes correctly split this into "ingestion" and the missing pieces. Let's build the whole picture end to end, in the order data actually flows.

### The big picture

```
Raw Documents → Load → Split (Chunk) → Embed → Store (Vector DB)   [INGESTION — done once/periodically]
                                                        ↓
User Query → Embed Query → Retrieve similar chunks → Augment Prompt → LLM → Answer   [RETRIEVAL — done per query]
```

RAG exists because an LLM's knowledge is frozen at training time and it can't know your private documents. RAG's job: **find the relevant pieces of your data and stuff them into the prompt** so the LLM answers using them instead of guessing.

---

### 19.1 Document Structure in LangChain

Everything loaded into LangChain becomes a `Document` object — this is the atomic unit you'll work with everywhere in the pipeline.

```python
from langchain_core.documents import Document

doc = Document(
    page_content="LangGraph lets you build stateful multi-agent workflows...",
    metadata={
        "source": "langgraph_notes.md",
        "chapter": 2,
        "author": "qasim",
        "date": "2026-07-01",
        "category": "framework-notes"
    }
)
```

Two fields, and both matter:

- **`page_content`** — the actual text that gets embedded and eventually shown to the LLM.
- **`metadata`** — a dict that does **not** get embedded, but travels alongside the chunk everywhere. This is what lets you **filter** at retrieval time.

**Why metadata matters (this is the part your notes flagged as unclear):** Say you have a RAG system over your blog posts. Without metadata filtering, a query about "LangGraph" might retrieve chunks from an unrelated post that happens to mention LangGraph once. With metadata, you can restrict retrieval:

```python
# At retrieval time, filter by metadata before/during similarity search
retriever = vectorstore.as_retriever(
    search_kwargs={
        "filter": {"category": "framework-notes"}  # only search within this category
    }
)
```

Metadata is also how you handle **access control** (only retrieve docs this user is allowed to see), **recency** (only retrieve docs from the last 30 days), and **source attribution** (show the user which document an answer came from).

---

### 19.2 Loaders — getting documents in

A **Loader** takes a raw source (PDF, website, database, folder) and turns it into a list of `Document` objects.

```python
from langchain_community.document_loaders import (
    PyPDFLoader,        # single PDF
    WebBaseLoader,      # a webpage URL
    DirectoryLoader,    # a whole folder of files
    CSVLoader,          # CSV rows → documents
    TextLoader,         # plain .txt file
)

pdf_docs = PyPDFLoader("resume.pdf").load()
web_docs = WebBaseLoader("https://qasim-mehar.github.io/blog/mcp-basics").load()
folder_docs = DirectoryLoader("./notes", glob="**/*.md").load()
```

Every loader's `.load()` returns `list[Document]` — usually **one Document per page (PDF) or per file**, each carrying metadata the loader auto-fills (like `source` and `page` number). This consistent output is exactly why the rest of your pipeline (splitter → embedder → vector store) doesn't care which loader you used.

---

### 19.3 Chunking (Splitting) — types and why it matters

You can't embed a 50-page PDF as one vector — it's too long, and the embedding would be a vague average of everything, useless for precise retrieval. So you **split** documents into smaller chunks first.

**The core tradeoff:** chunks too small → lose context (a chunk with just "the score was 85" means nothing without knowing *what* scored 85). Chunks too large → each chunk covers too many unrelated ideas, diluting the embedding and burying the relevant sentence among irrelevant ones.

| Splitter | How it splits | Best for |
|---|---|---|
| `CharacterTextSplitter` | Splits on a fixed character count | Simple, rarely ideal alone |
| `RecursiveCharacterTextSplitter` | Tries paragraph → sentence → word → character breaks, in order, to keep chunks semantically whole | **Default choice for most text** |
| `MarkdownHeaderTextSplitter` | Splits by `#`/`##` headers | Docs like yours — keeps each chapter/section intact |
| `RecursiveJsonSplitter` | Splits nested JSON while preserving structure | Structured/API data |
| `SemanticChunker` | Uses embeddings to split where *meaning* actually shifts, not just character count | Highest quality, more expensive to compute |

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # target size per chunk, in characters
    chunk_overlap=150,    # overlap between consecutive chunks
)
chunks = splitter.split_documents(pdf_docs)
```

**Why `chunk_overlap` exists:** if a key sentence gets cut exactly at a chunk boundary, overlap ensures it still appears whole in at least one chunk, so you don't lose meaning right at the seam.

For your notes specifically, `MarkdownHeaderTextSplitter` would be the smart pick — split by `##` so each Chapter becomes its own retrievable unit, preserving the "Chapter 16: Streaming" heading as metadata on every chunk from that section.

```python
from langchain.text_splitter import MarkdownHeaderTextSplitter

headers_to_split_on = [("##", "chapter")]
md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
chunks = md_splitter.split_text(your_markdown_notes)
# each chunk.metadata["chapter"] = "Chapter 16: Streaming — ..."
```

---

### 19.4 Embeddings — turning text into vectors

An **embedding model** converts text into a fixed-length list of numbers (a vector) such that texts with similar *meaning* end up as vectors that are close together in that space. This is the mathematical trick that makes "search by meaning" instead of "search by exact keyword" possible.

```python
from langchain_mistralai import MistralAIEmbeddings

embedder = MistralAIEmbeddings(model="mistral-embed")
vector = embedder.embed_query("What is a reducer in LangGraph?")
# vector = [0.021, -0.114, 0.083, ...] — e.g. 1024 numbers
```

Same embedding model must be used for both indexing your documents **and** embedding the user's query later — mixing embedding models breaks similarity search, since different models place meaning in different vector spaces.

---

### 19.5 Vector Databases — storing and searching vectors

A vector DB stores each chunk's embedding (plus its text and metadata) and lets you ask "give me the top-k chunks closest to this query vector" using similarity math (commonly cosine similarity).

```python
from langchain_chroma import Chroma

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedder,
    persist_directory="./chroma_db"   # saved to disk, not lost on restart
)

# Later — turn it into a retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})  # top 4 matches
```

Common choices: **Chroma** (simple, local, great for learning/small projects — good pick for OrchestraAI right now), **FAISS** (fast, local, no server needed), **Pinecone/Qdrant/Weaviate** (managed, scale to millions of vectors, production-grade).

---

### 19.6 Retrieval — the actual search step

At query time: embed the user's question with the *same* embedder, then ask the vector store for the most similar chunks.

```python
relevant_chunks = retriever.invoke("How does the tools_condition router work?")
for doc in relevant_chunks:
    print(doc.page_content[:100], doc.metadata)
```

Beyond plain similarity search, worth knowing:
- **MMR (Maximal Marginal Relevance)** — retrieves relevant *and* diverse chunks, avoiding 4 near-duplicate results.
- **Hybrid search** — combines vector similarity with traditional keyword (BM25) search, useful when exact terms (like error codes, function names) matter as much as meaning.
- **Reranking** — retrieve a wider net (say top 20) with the vector store, then use a smaller specialized reranker model to re-sort down to the best 4 — improves precision.

---

### 19.7 Augmentation — the "A" in RAG

This is the step your notes asked about directly: **augmentation = injecting retrieved chunks into the prompt before it goes to the LLM.**

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""
Answer the question using ONLY the context below. If the answer isn't in the context, say you don't know.

Context:
{context}

Question:
{question}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# The full RAG chain, LCEL-style (you already know this pattern from LangChain)
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

answer = rag_chain.invoke("What does MemorySaver do?")
```

That's the whole loop: **retrieve → format into context → fill the prompt template → send to LLM → parse output.** Everything before this (loaders, chunking, embeddings, vector store) exists purely to make this one step possible.

### 19.8 RAG as a LangGraph node (tying it back to what you know)

Since you already think in graphs, the natural way to wire RAG into an agent is as a **tool** the agent can call, not a fixed pipeline:

```python
from langchain_core.tools import tool

@tool
def search_notes(query: str) -> str:
    """Search Qasim's personal AI engineering notes for relevant context."""
    docs = retriever.invoke(query)
    return format_docs(docs)

# Now this is just another tool, exactly like your web_search tool
tools = [search_notes]
agent = create_react_agent(model, tools)
```

This is **agentic RAG** — instead of always retrieving on every query, the LLM *decides* when it needs to search your knowledge base, same as it decides when to call any other tool.

---

## ✅ Updated Summary Table

| Concept                                                      | Learned? | Notes                                           |
| --------------------------------------------------------------| ----------| -------------------------------------------------|
| `stream` vs `astream`                                        | ✅        | Sync for scripts, async for real backends       |
| `stream_mode` (values/updates/messages/debug/custom)         | ✅        | Controls chunk shape, not just streaming        |
| `interrupt()`                                                | ✅        | Pauses a node, needs a checkpointer             |
| `Command` (resume + goto)                                    | ✅        | Resume paused graphs; also general routing tool |
| HITL pattern                                                 | ✅        | approve-before-execute graphs                   |
| MCP transports (stdio, Streamable HTTP, deprecated SSE)      | ✅        | Only 2 official now                             |
| `langchain-mcp-adapters`                                     | ✅        | Converts MCP tools → LangChain tools            |
| MCP 2026 changes (stateless core, MCP Apps, Tasks extension) | ✅        | RC now, final spec July 28 2026                 |
| Document structure & metadata filtering                      | ✅        | `page_content` + `metadata`                     |
| Loaders                                                      | ✅        | PDF, Web, Directory, CSV, Text                  |
| Chunking types                                               | ✅        | Recursive, Markdown-header, Semantic            |
| Embeddings                                                   | ✅        | Text → vector, same model for docs & queries    |
| Vector DBs                                                   | ✅        | Chroma/FAISS (local), Pinecone/Qdrant (managed) |
| Retrieval (+ MMR, hybrid, reranking)                         | ✅        | Similarity search and its upgrades              |
| Augmentation                                                 | ✅        | Context injection into prompt template          |
| Agentic RAG (RAG as a tool)                                  | ✅        | Ties back to your existing agent-loop knowledge |

## ✅ Summary Table: What You Have Learned

| Concept                            | Learned? | Project              |
| ------------------------------------| ----------| ----------------------|
| Gen AI vs Agentic AI               | ✅        | Theory               |
| What is LangGraph                  | ✅        | Theory               |
| Why LangGraph vs LangChain         | ✅        | Theory               |
| Sequential Workflow                | ✅        | script_writter       |
| Parallel Workflow (Fan-Out/Fan-In) | ✅        | AI_content_Moderator |
| Conditional Workflow               | ✅        | chat_bot             |
| Reducers (custom + add_messages)   | ✅        | Both                 |
| TypedDict State                    | ✅        | All projects         |
| Annotated (metadata on state)      | ✅        | All projects         |
| Router Function                    | ✅        | chat_bot             |
| Conditional Edges                  | ✅        | chat_bot             |
| ToolNode                           | ✅        | chat_bot             |
| @tool decorator                    | ✅        | chat_bot             |
| bind_tools                         | ✅        | chat_bot             |
| tools_condition                    | ✅        | chat_bot             |
| MemorySaver                        | ✅        | chat_bot             |
| Thread ID (session memory)         | ✅        | chat_bot             |
| Streaming (stream_mode)            | ✅        | chat_bot             |
| .env / dotenv setup                | ✅        | All projects         |
| Virtual environments               | ✅        | All projects         |
| LangSmith (tracing)                | ❌        | Not yet              |
| Persistent Checkpointers           | ❌        | Not yet              |
| Structured Output (Pydantic)       | ❌        | Not yet              |
| Multi-Agent Systems                | ❌        | Not yet              |
| Subgraphs                          | ❌        | Not yet              |

---

> 💪 **You have covered an impressive range for a beginner. The foundation is solid. Keep building!**

