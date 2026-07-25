# My Agentic AI Learning Notes

> Notes from building, not just reading. Every project here taught something specific, and this file is organized so each concept builds on the one before it instead of jumping around.

---

# Part I: Foundations

## Chapter 1: Gen AI vs Agentic AI

Generative AI generates content, text, images, audio, based on a prompt. You give it input, it gives you output. One call, one response. ChatGPT answering a question is this: type, get an answer, done.

The limitation is that it's stateless and passive. It does what you ask and nothing more. It can't plan across steps, can't decide to use a tool on its own, and forgets everything the moment the response ends.

Agentic AI is a system that reasons, plans, decides, and acts across multiple steps to reach a goal, instead of answering once and stopping.

| Feature | Gen AI | Agentic AI |
|---|---|---|
| Takes one step | Yes | Yes |
| Plans multiple steps | No | Yes |
| Uses tools (search, code, APIs) | No | Yes |
| Makes decisions based on output | No | Yes |
| Has memory across turns | No | Yes |
| Can loop or branch logic | No | Yes |

Gen AI is a smart answering machine. Agentic AI is closer to an employee: it figures out what to do next, pulls in the tools it needs, and keeps working until the job is actually finished.

---

## Chapter 2: What Is LangGraph?

LangGraph is a Python framework built on top of LangChain for building stateful, multi-step AI applications, structured as a **graph**.

A graph, in the computer science sense, is a set of **nodes** (steps or functions) connected by **edges** (the paths data flows along):

```
START → Node A → Node B → Node C → END
```

Every node receives the current **state**, does some work (usually an LLM call), and returns an update to that state. The graph moves the data from node to node until it hits `END`.

### Why not just LangChain?

LangChain is a toolkit: models, tools, parsers, chains. It's built for linear flows, one call into the next. The moment you need branching, looping, or state that multiple steps read and write to, plain LangChain gets messy fast.

LangGraph adds an actual graph execution engine on top:

- Explicit state management, defined with a schema
- Conditional routing (decide at runtime which node runs next)
- Parallel execution (multiple nodes at once)
- Checkpointing (save and resume state between calls)
- Built-in tool execution via `ToolNode`

Rule of thumb: if the app has more than one step, or needs to make a decision at any point, reach for LangGraph.

---

# Part II: Core LangGraph Mechanics

## Chapter 3: Workflow Types

A workflow is the pattern in which nodes are connected. There are three you'll use constantly.

### 3.1 Sequential

Each node runs after the previous one finishes. Node A's output becomes Node B's input.

```
START → TextEditor → ScriptWriter → Translator → END
```

This is the `script_writter` project: `TextEditor` cleans up raw input, `ScriptWriter` turns it into a full script, `Translator` converts it to Roman Urdu. Each step depends on the last, so nothing here can run in parallel. It's also the easiest pattern to read and debug, which is why it's usually the right starting point.

```python
graph.add_edge(START, "TextEditor")
graph.add_edge("TextEditor", "ScriptWritter")
graph.add_edge("ScriptWritter", "Translator")
graph.add_edge("Translator", END)
```

### 3.2 Parallel (Fan-Out / Fan-In)

Multiple nodes run at the same time, then their results get collected back together.

```
         → Node A →
START →  → Node B →  → END
         → Node C →
```

The `AI_content_Moderator` project runs three checks, toxicity, cultural sensitivity, and copyright risk, on the same input **simultaneously**. Running them one after another would take three times as long for no benefit, since none of them depends on another's output.

```python
# Fan-out from START
graph_builder.add_edge(START, "toxicity_node")
graph_builder.add_edge(START, "cultural_node")
graph_builder.add_edge(START, "copyright_node")

# Fan-in to END
graph_builder.add_edge("toxicity_node", END)
graph_builder.add_edge("cultural_node", END)
graph_builder.add_edge("copyright_node", END)
```

### 3.3 Conditional

The graph decides which node runs next based on what the previous node produced. This is the pattern that makes an AI system feel "agentic" rather than scripted.

```
START → LLM Node → [router function] → Node A (if condition X)
                                     → Node B (if condition Y)
                                     → END
```

The `chat_bot` project uses this: after the LLM responds, `tools_condition` checks whether it asked to call a tool. If yes, route to the `tools` node. If no, route to `END`. After the tool runs, the graph loops back to the LLM. This loop is what people mean by an **agent loop**.

```python
graph_builder_with_tools.add_conditional_edges("chatbotWithTools", tools_condition)
graph_builder_with_tools.add_edge("tools", "chatbotWithTools")
```

---

## Chapter 4: State

State is the data container that flows through the whole graph. Every node reads from it and writes back to it, so think of it as the working memory of a single run.

Define it with `TypedDict`:

```python
from typing import TypedDict

class MyState(TypedDict):
    user_input: str
    result: str
```

A node receives the full state but returns only the fields it changed:

```python
def my_node(state: MyState) -> dict:
    text = state["user_input"]
    # do work
    return {"result": "some answer"}
```

LangGraph merges that partial return back into state automatically. You never manually reconstruct the whole dict.

---

## Chapter 5: Reducers

By default, when a node returns a value, LangGraph **replaces** whatever was already in that state field. Fine for a string. A problem for anything that needs to accumulate.

Say Node A writes `{"score": {"toxicity": 80}}` and Node B writes `{"score": {"culture": 40}}`. Without a reducer, Node B's write overwrites Node A's, and the toxicity score is gone.

A **reducer** is a function that defines how a new update should be *combined* with the existing value instead of replacing it.

```python
def merge_score_dicts(existing: dict, new_update: dict) -> dict:
    if existing is None:
        return new_update
    return {**existing, **new_update}
```

Attach it to a state field with `Annotated`:

```python
from typing import Annotated

class AnalyzerState(TypedDict):
    raw_text: str
    safety_score: Annotated[dict[str, int], merge_score_dicts]
```

Now three parallel nodes can all write to `safety_score`, and instead of clobbering each other, their results merge:

```json
{
    "toxicity": 85,
    "cultural_insensitivity": 20,
    "copyright_risk": 10
}
```

The reducer you'll use most often ships with LangGraph itself:

```python
from langgraph.graph import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]
```

`add_messages` appends to the list instead of overwriting it. This one line is the entire reason a chatbot remembers the conversation instead of losing history on every turn.

---

## Chapter 6: Router Functions and Conditional Edges

A router function reads the current state and returns a string, the name of the next node (or `END`).

```python
def my_router(state: State) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END
```

Wire it in with `add_conditional_edges`:

```python
graph.add_conditional_edges(
    "my_llm_node",
    my_router,
    {
        "tools": "tools_node",
        END: END
    }
)
```

LangGraph ships a ready-made router for the most common case, checking whether the LLM asked for a tool call:

```python
from langgraph.prebuilt import ToolNode, tools_condition

graph_builder.add_conditional_edges("chatbotWithTools", tools_condition)
```

`tools_condition` routes to `"tools"` if the last message contains a tool call, and to `END` otherwise. You almost never need to write this one by hand.

---

## Chapter 7: ToolNode and Tools

`ToolNode` is a prebuilt node that executes tools for you, so you don't hand-write the "call the function, capture the result, put it back in state" logic yourself.

```python
from langgraph.prebuilt import ToolNode

tools = [web_search]
tool_node = ToolNode(tools)
graph.add_node("tools", tool_node)
```

When the LLM asks to call `web_search(topic='AI')`, `ToolNode` intercepts that request, runs your actual function, and appends the result to state as a `ToolMessage`.

### Defining a tool

```python
from langchain.tools import tool

@tool
def web_search(topic: str) -> str:
    """Search on web for latest information"""
    return tavily_search.invoke(topic)
```

The docstring isn't a formality here, it's the only description the LLM has of when and how to use the tool. A vague docstring produces a tool the model calls at the wrong times or not at all.

### Binding tools to an LLM

The model has to know a tool exists before it can decide to call it:

```python
llm_with_tools = llm.bind_tools(tools)
```

---

## Chapter 8: Memory and Checkpointing

By default, every `graph.invoke()` call starts from nothing. That's fine for a one-shot pipeline like `script_writter`, but useless for a chatbot that needs to remember what you said two messages ago.

### MemorySaver

`MemorySaver` is LangGraph's in-memory checkpointer. It saves the full graph state after every step, keyed by a **thread ID**.

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)
```

### Thread ID

A `thread_id` works like a session ID. Invoke the graph twice with the same thread ID, and the second call picks up exactly where the first left off.

```python
config = {"configurable": {"thread_id": "1"}}

graph.invoke({"messages": [{"role": "user", "content": "my name is qasim"}]}, config=config)

# Remembers "qasim" because the thread_id matches
graph.invoke({"messages": [{"role": "user", "content": "what is my name?"}]}, config=config)
```

One caveat worth flagging now, since it matters later: `MemorySaver` lives in RAM. Restart the process and every conversation is gone. For anything real, you want a database-backed checkpointer like `SqliteSaver` or `PostgresSaver`, which use the exact same interface but persist to disk.

---

## Chapter 9: Streaming

Waiting for the entire graph to finish before showing anything to the user feels slow, especially for a chatbot. Streaming sends output as each step completes instead.

### `stream` vs `astream`

- `graph.stream(...)` is synchronous. It blocks the process while running. Fine for scripts and notebooks.
- `graph.astream(...)` is asynchronous, called with `await` inside an `async def`. This is what a real backend needs (FastAPI, for instance), because a long-running agent call shouldn't block every other request the server is handling.

```python
# Sync
for chunk in graph.stream({"messages": [...]}, config=config):
    print(chunk)

# Async, inside a FastAPI endpoint
async def run_agent(user_input: str, config: dict):
    async for chunk in graph.astream({"messages": [{"role": "user", "content": user_input}]}, config=config):
        yield chunk
```

Script or notebook: `stream`. Server handling multiple users: `astream`.

### `stream_mode`: what shape each chunk takes

This is easy to skim past, but it controls the *content* of what you get back, not just whether it streams.

| Mode | What you get | Typical use |
|---|---|---|
| `"values"` | Full current state after each node | Simple chatbots, you want the whole picture every time |
| `"updates"` | Only the dict the just-finished node returned | Debugging, or large state where you only care about the delta |
| `"messages"` | Token-by-token LLM output, plus which node/model produced it | A live typing effect in a UI |
| `"debug"` | Verbose internal events: task start, task end, checkpoints | Tracing exactly what the graph did, step by step |
| `"custom"` | Whatever you push manually via `get_stream_writer()` | Progress updates from inside a long tool call |

```python
# Full state snapshot each time
for chunk in graph.stream(inputs, config, stream_mode="values"):
    print(chunk["messages"][-1])

# Only the delta
for chunk in graph.stream(inputs, config, stream_mode="updates"):
    print(chunk)  # {"chatbotWithTools": {"messages": [...]}}

# Token streaming
for msg_chunk, metadata in graph.stream(inputs, config, stream_mode="messages"):
    print(msg_chunk.content, end="", flush=True)

# Multiple modes at once
for mode, chunk in graph.stream(inputs, config, stream_mode=["updates", "messages"]):
    ...
```

Rough mental model: `"values"` hands you the whole page after every edit, `"updates"` hands you just the sentence someone added, `"messages"` hands you individual pen strokes as they happen.

---

## Chapter 10: Human-in-the-Loop, `interrupt`, and `Command`

Some agent actions shouldn't happen without a human checking first: sending an email, deleting a record, spending money. Human-in-the-loop (HITL) is the pattern for pausing a graph right before an action like that and waiting for approval.

### `interrupt()`

Call `interrupt()` inside a node to pause execution at that exact point. Whatever you pass to it gets surfaced to whoever is running the graph.

```python
from langgraph.types import interrupt, Command

def human_approval_node(state: State) -> dict:
    decision = interrupt({
        "question": "Send this email?",
        "draft": state["draft_email"]
    })
    # Execution pauses on the line above until the graph is resumed
    if decision == "approve":
        return {"status": "approved"}
    return {"status": "rejected"}
```

This requires a checkpointer. Pausing means LangGraph has to save the exact state somewhere and be able to resume it later, possibly minutes or days later, so `MemorySaver` (or a persistent equivalent) is a prerequisite, not optional.

### Resuming with `Command`

You resume a paused graph by invoking it again with `Command(resume=...)` instead of normal input:

```python
config = {"configurable": {"thread_id": "1"}}

result = graph.invoke({"messages": [...]}, config=config)
print(result["__interrupt__"])  # the payload passed to interrupt()

result = graph.invoke(Command(resume="approve"), config=config)
```

`Command` is more general than just resuming, though. A node can return one to bundle a state update with an explicit "go to this node next" instruction, skipping `add_conditional_edges` entirely when the routing logic is simple:

```python
def router_node(state: State) -> Command:
    if state["needs_review"]:
        return Command(update={"status": "pending"}, goto="human_review")
    return Command(update={"status": "auto_approved"}, goto="execute")
```

### The pattern in practice

```
START → draft_email_node → human_approval_node (interrupt) → [approved] → send_email_node → END
                                                              → [rejected] → END
```

Anywhere an agent's action has a real-world, hard-to-undo effect, this is the shape to reach for: pause immediately before the irreversible step.

---

# Part III: Applying It

## Chapter 11: Project Deep Dives

### `script_writter`, sequential pipeline

Takes a rough idea in plain English and turns it into a polished Roman Urdu YouTube script.

```python
class pipleineState(TypedDict):
    raw_input: str
    edited_input: str
    script_text: str
    roman_urdu: str
```

Three nodes: `text_editor_node` fixes grammar and phrasing, `script_writter_node` writes the full script with hook, body, CTA, and visual directions, `roman_urdu_node` translates the narration (keeping visual directions in English).

What this project taught: defining and using `TypedDict` state, chaining nodes with `add_edge`, and using a detailed system prompt to control output format.

### `chat_bot`, conditional agent with tools and memory

A chatbot that searches the web and remembers the conversation.

```python
class State(TypedDict):
    messages: Annotated[list, add_messages]
```

The agent loop:

```
chatbotWithTools
    → (tool call needed) tools node → back to chatbotWithTools
    → (answer ready) END
```

This project is where most of the core mechanics from Part II come together at once: `add_messages`, `bind_tools`, `ToolNode`, `tools_condition`, `MemorySaver`, `thread_id`, and `graph.stream()`.

### `AI_content_Moderator`, parallel fan-out/fan-in

Scores a piece of text simultaneously for toxicity, cultural insensitivity, and copyright risk (0 to 100 each).

```python
class AnalyzerState(TypedDict):
    raw_text: str
    safety_score: Annotated[dict[str, int], merge_score_dicts]
```

Three nodes run in parallel: `toxicity_checker_node`, `cultural_sensitivity_node`, `copyright_checker_node`. The custom reducer, `merge_score_dicts`, is the actual point of this project, it's what lets all three write to the same field without one erasing another's result.

```json
{
    "toxicity": 85,
    "cultural_insensitivity": 20,
    "copyright_risk": 10
}
```

---

## Chapter 12: Tools and Libraries

| Library | Purpose |
|---|---|
| `langgraph` | Graph-based agentic workflow engine |
| `langchain` | LLM toolkit: tools, prompts, parsers |
| `langchain_mistralai` | Mistral AI LLM integration |
| `langchain_tavily` | Tavily web search tool |
| `python-dotenv` | Load API keys from `.env` files |
| `rich` | Formatted terminal output |
| `typing.TypedDict` | Structured state schemas |
| `typing.Annotated` | Attach reducer metadata to state fields |

---

## Chapter 13: Project Setup Patterns

Every project here follows the same setup, worth internalizing once rather than re-deriving each time.

**Virtual environment**, isolates dependencies per project:
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows PowerShell
```

**Dependencies**, listed in `pyproject.toml` or `requirements.txt`:
```bash
pip install -r requirements.txt
# or
uv sync
```

**`.env` file**, stores API keys, never committed to Git:
```
MISTRAL_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
```

**`.gitignore`**, keeps `.env`, `.venv`, `__pycache__` out of version control.

---

## Chapter 14: The Big Picture

```
Your Idea (raw input)
       ↓
   StateGraph(YourState)    ← state holds all your data
       ↓
   Nodes (functions)        ← each node does one job
       ↓
   Edges                    ← sequential, parallel, or conditional
       ↓
   Reducers                 ← merge updates from parallel nodes
       ↓
   Checkpointer             ← memory across calls, plus HITL pauses
       ↓
   Final State = Result
```

Every project above is a variation on this exact shape. More complex use cases just combine more of these pieces.

---

# Part IV: MCP

## Chapter 15: Model Context Protocol

### The problem it solves

In plain LangChain, connecting an agent to Gmail means writing a `@tool`-decorated function that calls the Gmail API, with a schema, bound to the LLM. Another team building a different agent, maybe a different framework, maybe a different language, wants Gmail too, so they write their own Gmail integration from scratch. Every team reimplements the same wiring.

MCP standardizes this. A tool gets exposed once by an independent server, and any MCP-compatible client can talk to it without custom glue code per integration. The common comparison is USB-C for AI tools: one interface, many devices on either end.

### The three roles

1. **Host**, the application the user interacts with (your LangGraph app, Claude Desktop)
2. **Client**, lives inside the host, holds a 1:1 connection to one server
3. **Server**, exposes capabilities: tools, data, prompts

One host can run several clients at once, each talking to a different server. A single LangGraph agent might connect to a GitHub server, a Postgres server, and a Slack server simultaneously.

### The protocol

Underneath, it's JSON-RPC 2.0 over a transport. Client and server negotiate capabilities on connect, before any real calls happen.

| Transport | How it works | Use case |
|---|---|---|
| **STDIO** | Client spawns the server as a local subprocess, talks over stdin/stdout | Local tools: filesystem access, a local script, a database on the same machine |
| **Streamable HTTP** | Client sends HTTP requests to a server URL; server can reply with one JSON response or open a stream | Remote/hosted servers, a SaaS wrapping their API as MCP |
| **HTTP+SSE** *(deprecated)* | A separate SSE stream for server-to-client, HTTP POST for client-to-server | Legacy only. Streamable HTTP replaced it because a persistent SSE connection complicated infrastructure. Don't build new servers on this. |

### The three primitives

- **Tools**, model-controlled. The LLM decides when to call them, same idea as a `@tool` function. `list_tools()` / `call_tool()`.
- **Resources**, application-controlled. The host app decides what to pull in, not the model autonomously, think context injection the app manages.
- **Prompts**, reusable prompt templates a server ships alongside its tools, so it can recommend "here's the pattern for using this" and not just "here's a function."

Most people only touch Tools at first. Resources and Prompts are underused.

### Connecting it to LangGraph

`langchain-mcp-adapters` converts MCP tools into LangChain `Tool` objects, so `create_react_agent`, `create_agent`, or a custom `ToolNode` can use them exactly like any hand-written tool.

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient({
    "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/qasim/projects"],
        "transport": "stdio",
    },
    "github": {
        "url": "https://api.githubcopilot.com/mcp/",
        "transport": "streamable_http",
        "headers": {"Authorization": "Bearer <token>"},
    },
})

tools = await client.get_tools()
```

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(model, tools)
result = await agent.ainvoke({"messages": [{"role": "user", "content": "list files in my repo"}]})
```

### What's changing in 2026

The stable spec as of this writing is still `2025-11-25`, and that's what production servers should target. A release candidate (`2026-07-28`) is close to final and is the largest revision since MCP launched:

- **Going stateless.** Previously a Streamable HTTP client did an `initialize` handshake and got back an `Mcp-Session-Id` that every later request had to carry, meaning servers needed sticky sessions and shared session stores to scale. The RC drops the handshake and session ID entirely, so any request can hit any server instance behind a plain load balancer.
- **MCP Apps**, servers can ship interactive HTML UIs rendered in a sandboxed iframe inside the host, instead of returning plain text or JSON only.
- **Tasks** moved from a core feature into an official extension for long-running work, `tasks/get`, `tasks/update`, `tasks/cancel` replace one blocking call.
- **Roots, Sampling, and Logging are deprecated** (not removed, minimum 12-month window) in favor of tool parameters, resource URIs, direct provider APIs, and standard telemetry.
- **Authorization hardening**, closer alignment with OAuth 2.1 and OpenID Connect, including an enterprise SSO extension for managing access to many servers at once.

Practical takeaway: build against the stable `2025-11-25` spec today, that's what `langchain-mcp-adapters` and most existing servers target. The stateless behavior will show up in servers over the next few months, expected, not a bug.

---

# Part V: RAG

## Chapter 16: The RAG Pipeline, Overview

RAG exists because an LLM's knowledge is frozen at training time and it has no access to your private documents. The job of RAG is to find the relevant pieces of your data and put them in the prompt, so the model answers from what's actually there instead of guessing.

```
Raw Documents → Load → Split (Chunk) → Embed → Store (Vector DB)   [ingestion, done once or periodically]
                                                        ↓
User Query → Embed Query → Retrieve similar chunks → Augment Prompt → LLM → Answer   [retrieval, done per query]
```

Everything below follows that order: how documents are structured, how they're loaded, how they're split, how they're embedded and stored, how they're retrieved, and finally how retrieval results get folded into a prompt.

---

## Chapter 17: Document Structure and Metadata

Everything loaded into LangChain becomes a `Document`, the atomic unit for the rest of the pipeline.

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

Two fields matter here:

- **`page_content`**, the text that gets embedded and eventually shown to the LLM.
- **`metadata`**, a dict that does not get embedded but travels with the chunk everywhere. This is what makes filtering at retrieval time possible.

Without metadata, a query about "LangGraph" might pull a chunk from an unrelated document that happens to mention it once in passing. With metadata, you can scope the search:

```python
retriever = vectorstore.as_retriever(
    search_kwargs={
        "filter": {"category": "framework-notes"}
    }
)
```

Metadata is also how access control (only retrieve documents this user can see), recency (only the last 30 days), and source attribution (show which document an answer came from) get implemented.

---

## Chapter 18: Loaders

A loader takes a raw source (PDF, website, database, folder) and turns it into a list of `Document` objects.

```python
from langchain_community.document_loaders import (
    PyPDFLoader,        # single PDF
    WebBaseLoader,      # a webpage URL
    DirectoryLoader,    # a whole folder of files
    CSVLoader,          # CSV rows to documents
    TextLoader,         # plain .txt file
)

pdf_docs = PyPDFLoader("resume.pdf").load()
web_docs = WebBaseLoader("https://qasim-mehar.github.io/blog/mcp-basics").load()
folder_docs = DirectoryLoader("./notes", glob="**/*.md").load()
```

Every loader's `.load()` returns `list[Document]`, typically one per page for a PDF or one per file, with metadata auto-filled (`source`, `page`, and so on). That consistent shape is exactly why the rest of the pipeline, splitter, embedder, vector store, doesn't need to know which loader produced the input.

---

## Chapter 19: Chunking

A 50-page PDF can't be embedded as a single vector, it's too long, and the resulting embedding would be a vague average of everything in it, useless for precise retrieval. So documents get split into smaller chunks first.

The core tradeoff: chunks too small lose context (a chunk that just says "the score was 85" means nothing without knowing what scored 85). Chunks too large dilute the embedding, burying the one relevant sentence among unrelated ones.

| Splitter | How it splits | Best for |
|---|---|---|
| `CharacterTextSplitter` | Fixed character count | Simple text, rarely ideal alone |
| `RecursiveCharacterTextSplitter` | Tries paragraph, then sentence, then word, then character breaks in order | Default choice for most text |
| `MarkdownHeaderTextSplitter` | Splits on `#` / `##` headers | Structured docs like these notes, keeps each section intact |
| `RecursiveJsonSplitter` | Splits nested JSON while preserving structure | Structured/API data |
| `SemanticChunker` | Uses embeddings to split where meaning actually shifts | Highest quality, more expensive to compute |

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
)
chunks = splitter.split_documents(pdf_docs)
```

`chunk_overlap` exists so that a sentence landing exactly on a chunk boundary still appears whole in at least one chunk, instead of getting cut in half at the seam.

For a document like this one, `MarkdownHeaderTextSplitter` is the better fit, splitting on `##` keeps each chapter as its own retrievable unit and carries the chapter title into the metadata automatically:

```python
from langchain.text_splitter import MarkdownHeaderTextSplitter

headers_to_split_on = [("##", "chapter")]
md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
chunks = md_splitter.split_text(your_markdown_notes)
# each chunk.metadata["chapter"] = "Chapter 16: ..."
```

---

## Chapter 20: Embeddings

An embedding model converts text into a fixed-length vector such that texts with similar meaning land close together in that vector space. This is what makes "search by meaning" possible instead of exact keyword matching.

```python
from langchain_mistralai import MistralAIEmbeddings

embedder = MistralAIEmbeddings(model="mistral-embed")
vector = embedder.embed_query("What is a reducer in LangGraph?")
# [0.021, -0.114, 0.083, ...]
```

The same embedding model has to be used for both indexing documents and embedding the query later. Different embedding models place meaning in different vector spaces, so mixing them breaks similarity search silently, it won't error, it'll just return bad matches.

---

## Chapter 21: Vector Databases

A vector database stores each chunk's embedding alongside its text and metadata, and answers queries like "give me the top-k chunks closest to this vector," usually via cosine similarity.

```python
from langchain_chroma import Chroma

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedder,
    persist_directory="./chroma_db"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
```

Common choices: Chroma and FAISS for local, no-server setups (a good default for a personal project), Pinecone, Qdrant, or Weaviate when you need managed infrastructure at millions-of-vectors scale.

---

## Chapter 22: Retrieval

At query time, embed the question with the same embedder used for the documents, then ask the vector store for the closest matches.

```python
relevant_chunks = retriever.invoke("How does the tools_condition router work?")
for doc in relevant_chunks:
    print(doc.page_content[:100], doc.metadata)
```

Beyond plain similarity search:

- **MMR (Maximal Marginal Relevance)**, retrieves results that are both relevant and diverse, so you don't get four near-duplicate chunks.
- **Hybrid search**, combines vector similarity with keyword search (BM25), useful when exact terms, error codes, function names, matter as much as meaning.
- **Reranking**, retrieve a wide net (say top 20) with the vector store, then use a smaller specialized model to re-sort down to the best 4. Improves precision at the cost of an extra step.

---

## Chapter 23: Augmentation

Augmentation is the step where retrieved chunks get injected into the prompt before it reaches the LLM, the "A" in RAG.

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

That's the entire loop: retrieve, format into context, fill the prompt template, send to the LLM, parse the output. Everything in Chapters 17 to 22 exists only to make this one step work.

### RAG as a LangGraph tool

Since you already think in graphs, the natural way to wire RAG into an agent is as a tool the agent can choose to call, not a fixed pipeline that always runs:

```python
from langchain_core.tools import tool

@tool
def search_notes(query: str) -> str:
    """Search Qasim's personal AI engineering notes for relevant context."""
    docs = retriever.invoke(query)
    return format_docs(docs)

tools = [search_notes]
agent = create_react_agent(model, tools)
```

This is **agentic RAG**: instead of retrieving on every single query, the LLM decides *when* it needs to search the knowledge base, exactly the same decision process it uses for any other tool.

---

## Chapter 24: Traditional RAG, Strengths and Limits

Worth being honest about both sides before assuming vector-based RAG is always the right architecture.

**What it's good at:**
- Scales to millions of documents without much trouble
- Mature ecosystem, Chroma, FAISS, Pinecone, and the tooling around them are well understood
- Retrieval itself is cheap, a nearest-neighbor lookup is fast and inexpensive
- Strong for factoid-style questions, "what is X" queries with a clear, localized answer
- Domain-agnostic, works reasonably out of the box on almost any text

**Where it breaks down:**
- Chunking destroys context. Splitting a document into fixed windows can cut a sentence, a table, or a cross-reference right in half.
- Similarity is not the same as relevance. A chunk can be semantically close to the query wording and still not contain the answer.
- No cross-section reasoning. Vector search finds the single closest chunk; it doesn't naturally connect facts that live in different sections of a document.
- Hard to explain. There's no clean answer to "why did the system pick this chunk," beyond a similarity score.
- Embedding drift. Change the embedding model and every stored vector is now in a different space, meaning a full re-embed of the entire corpus.

---

## Chapter 25: Vectorless RAG

### The core idea

Vectorless RAG removes the embedding and vector similarity layer entirely. No embedding model, no vector database, no cosine similarity, no chunking into arbitrary windows. Instead, it builds a **hierarchical tree** of a document's actual structure (sections, subsections) and has an LLM reason over that tree to decide where the answer likely lives.

The comparison that makes this click: a human reading a 200-page financial report doesn't scan every paragraph for the one that's statistically closest to their question. They check the table of contents, jump to the right section, and read it. Vectorless RAG is that process, automated.

The best-known implementation is **PageIndex** (VectifyAI, released 2025), which frames retrieval as tree navigation rather than similarity search, an approach conceptually borrowed from how game-playing systems like AlphaGo use a learned strategy to search a space intelligently instead of exhaustively.

### The pipeline

```
Document → Parse into hierarchical tree (sections, subsections, with summaries per node)
        → Query arrives
        → LLM reads the tree (titles + summaries, like a table of contents)
        → LLM reasons about which node(s) likely contain the answer
        → Retrieve full text of those nodes
        → Generate answer, with a reasoning trace showing exactly which nodes were chosen and why
```

The key difference from vector RAG happens at the retrieval step. A vector database computes similarity scores for every chunk in parallel and returns the top matches, no reasoning involved. A tree-based system asks the LLM directly: given this document's structure and this question, where should I look? That lets it follow cross-references, recognize a question about an appendix belongs in the appendix node, or split a multi-part question across two different sections, none of which a similarity score alone can do.

### How it differs from traditional RAG

|                     | Vector RAG                                                    | Vectorless (tree-based) RAG                                                                                      |
| ---------------------| ---------------------------------------------------------------| ------------------------------------------------------------------------------------------------------------------|
| Index               | Embeddings in a vector DB                                     | Hierarchical tree with per-node summaries                                                                        |
| Chunking            | Fixed-size windows, often arbitrary                           | Natural document sections                                                                                        |
| Retrieval mechanism | Cosine similarity, computed in parallel                       | LLM reasoning over structure, sequential                                                                         |
| Explainability      | Weak, a similarity score isn't a reason                       | Strong, full reasoning trace over which nodes were picked and why                                                |
| Latency             | Fast, roughly 200 to 500 ms                                   | Slower, roughly 2 to 4 seconds                                                                                   |
| Cost per query      | Low                                                           | Higher, more LLM tokens spent per query                                                                          |
| Best fit            | Large, loosely structured corpora, single-hop factoid queries | Structurally rich documents (contracts, policies, technical specs), multi-hop or cross-reference-heavy questions |

Reported accuracy gains are notable on the kind of dense, structured documents this method targets. PageIndex's own benchmark reports around 98.7% on FinanceBench, a financial-document QA benchmark, well above typical vector RAG baselines on the same kind of documents. That's a best-case number for a favorable document type, not a general claim that tree-based retrieval beats vector search everywhere.

### When to reach for which

Vectorless RAG fits when the corpus is structurally rich (policies, contracts, regulations, technical specs), when users need to audit *why* a particular section was retrieved, and when the corpus is small enough (tens of thousands of documents, not billions of web pages) that paying extra LLM tokens per query is affordable. Beyond that scale, or for single-hop semantic search over unstructured text, vector RAG is still the better default. Some production systems combine both: a tree index for coarse, section-level navigation, then vector search within the selected section for fine-grained retrieval.

---




# Part VI: What's Still Missing

## Chapter 26: Gaps to Close Next

Honest list of what hasn't been covered yet but matters for production agentic systems:

| Topic                          | What it is                                                                                                                                           |
| --------------------------------| ------------------------------------------------------------------------------------------------------------------------------------------------------|
| **LangSmith**                  | Tracing and debugging LangGraph runs, see exactly what each node did, what prompt was sent, what came back. Listed as a dependency but not yet used. |
| **Persistent checkpointers**   | `SqliteSaver` / `PostgresSaver`, the database-backed replacements for `MemorySaver` once memory needs to survive a restart.                          |
| **Subgraphs**                  | Embedding one LangGraph inside another, for large systems where different parts of an agent are their own isolated workflows.                        |
| **Multi-agent systems**        | Several specialized agents (researcher, writer, critic), each its own graph, coordinated by a supervisor.                                            |
| **Error handling and retries** | What happens when an LLM returns something malformed or an API call fails, and how to add retry/fallback logic to a node.                            |
| **Structured output**          | `llm.with_structured_output(MyPydanticModel)` instead of manually parsing `res.content.strip()`.                                                     |
| **Iterative workflows**        | Loops that keep refining an output until some quality bar is met, rather than running once.                                                          |
| **RAG evaluation**             | Measuring retrieval quality (precision/recall on retrieved chunks) and answer quality systematically, instead of eyeballing outputs.                 |

---

# Part VII: Deep Agents

## Chapter 27: Shallow Agents, the ReAct Loop, and Where It Breaks

### What a shallow agent actually is

Everything you built in `chat_bot`, the LLM node plus `ToolNode` plus `tools_condition` looping back on itself, is a **ReAct agent**. ReAct stands for Reason + Act: the model reasons about what to do, calls a tool, observes the result, reasons again, and repeats until it decides it's done.

```
LLM reasons → calls a tool → observes result → LLM reasons again → ... → done
```

This is the shape `create_react_agent` and `langchain.agents.create_agent` both wrap for you. It's usually called a **shallow agent** once you compare it to what's coming next in this chapter, not because it's badly built, but because everything it knows lives in one flat, ever-growing list of messages.

### Where the ReAct loop breaks down

The loop works well for short tasks: a handful of tool calls, a bounded conversation. It starts failing in specific, predictable ways once a task gets long or complex:

**Context window overflow.** Every tool call and every result gets appended to the same `messages` list. A research task that runs 40 tool calls means 40 tool results sitting in context, most of them irrelevant by the time the agent needs to write the final answer. Eventually you hit the model's context limit, or the model starts losing track of the actually important information buried under noise.

**No real planning.** A ReAct agent decides its next action one step at a time, based only on what just happened. It has no explicit plan it's working through and nothing that keeps it honest about what's left to do. Ask it to do a five-part task and it might nail three parts and quietly forget the other two, because there was never a persisted list of "these are the five things I need to finish."

**Nothing survives a compaction or a crash.** If you truncate old messages to save context space, whatever information lived only in that truncated history is gone. There's no separate place the agent could have written down an important intermediate finding to protect it.

**One flat context for every subtask.** A research agent gathering information from ten sources dumps all ten sources' worth of content into the same context the final writing step also has to work in. There's no way to isolate "the noisy work of finding source 7" from "the clean context needed to write the final report."

**No delegation.** Everything happens in one agent, one context, sequentially. There's no clean way to hand off an independent chunk of the task to a separate reasoning process and just get back a summary.

These aren't bugs you can patch by writing a better system prompt. They're structural limits of "one growing list of messages, one loop." Deep agents are the architectural fix.

---

## Chapter 28: What a Deep Agent Actually Is

A deep agent, in the sense LangChain uses the term with its `deepagents` library, is a ReAct-style loop wrapped with a specific set of built-in infrastructure: planning, a virtual filesystem, subagent delegation, and automatic context management. The name comes from the fact that it can go deep on a task instead of staying shallow and reactive.

The pitch is genuinely simple to use:

```python
from deepagents import create_deep_agent

agent = create_deep_agent(
    model=model,
    tools=[web_search],
    system_prompt="Act as a researcher",
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Research LangGraph and write a summary"}]
})
```

Compare that to the plain ReAct version:

```python
from langchain.agents import create_agent

simple_agent = create_agent(model=model, tools=[web_search])
```

Same three lines, same tools list, but `simple_agent` has no planning, no filesystem, no delegation. Everything from here in this chapter is what those extra capabilities inside `create_deep_agent` are actually doing.

It's built on LangGraph underneath, so everything from Part II still applies: streaming, checkpointing, human-in-the-loop, `interrupt()`. Deep Agents is a harness on top, not a replacement.

---

## Chapter 29: The Four Core Components

### 29.1 Planning, the `write_todos` tool

Every deep agent gets a built-in `write_todos` tool by default. When it receives a task with multiple parts, it writes an explicit todo list into its own state before starting, then updates that list as items get completed.

```
write_todos([
    "Search for LangGraph documentation",
    "Search for recent LangGraph release notes",
    "Summarize findings into summary.md"
])
```

This is the direct fix for "no real planning" from Chapter 27. The plan isn't implicit in the model's reasoning anymore, it's an explicit, persisted list the agent can check itself against, and that you can inspect from the outside to see exactly what the agent thinks it still needs to do.

### 29.2 Subagents, the `task` tool

A deep agent can spawn subagents to handle independent pieces of work, each subagent getting its own isolated context window. The parent agent calls a subagent through the built-in `task` tool, waits for it to finish, and receives back a summary rather than the subagent's entire raw working context.

```python
from deepagents import create_deep_agent

research_subagent = {
    "name": "researcher",
    "description": "Searches the web and returns findings on a topic",
    "prompt": "You are a research specialist. Search and summarize findings.",
    "tools": [web_search],
}

agent = create_deep_agent(
    model=model,
    tools=[web_search],
    subagents=[research_subagent],
)
```

This solves two of the Chapter 27 problems at once. It gives you delegation, an independent subtask runs in its own process instead of jamming everything into one sequential loop, and it gives you context isolation, ten noisy tool calls spent finding a source live only in the subagent's context, and the parent only ever sees the clean summary that comes back. Subagents can even run in parallel for genuinely independent subtasks.

### 29.3 The virtual filesystem, persistent and shared

Deep agents get a set of file tools (read, write, edit, search) backed by a **virtual filesystem**. This is the mechanism that solves "nothing survives a compaction or a crash" and "no shared memory across subagents" simultaneously.

The filesystem isn't tied to the actual disk by default, it's a pluggable backend, state-backed by default (lives in the LangGraph state/checkpointer), or swappable for a real disk, a database, or cloud storage. Crucially, it's accessible to the main agent *and* every subagent it spawns, which makes it the shared memory layer between them. A subagent can write a finding to a file, and the main agent (or a different subagent) can read it later, without that finding ever having lived in either agent's message history.

```python
result = agent.invoke({"messages": "Research LangGraph and write a summary in summary.md"})

# The files the agent created or touched are available in the result
result["files"]  # e.g. {"summary.md": "...", "notes/source_1.md": "..."}
```

### 29.4 The system prompt

The system prompt in a deep agent isn't just "be a helpful assistant", it's the layer that tells the agent *how* to use the other three components: when to write a todo list, when to delegate to a subagent instead of doing the work itself, when to write findings to a file instead of keeping them in the message history. `create_deep_agent`'s default prompt already encodes sensible defaults for this, and you extend it with `system_prompt=` for domain-specific behavior, as in the "Act as a researcher" example above.

---

## Chapter 30: Middleware, the Mechanism Behind All of This

None of the four components above are hardcoded into one big function. They're implemented as **middleware**, composable pieces that hook into the agent loop (before/after the model call, before/after a tool call) and modify its behavior. `create_deep_agent` bundles a specific default stack of middleware so you get all of it without wiring anything by hand:

- **Planning middleware**, adds the `write_todos` tool and the todo-tracking behavior
- **Filesystem middleware**, adds the file read/write/edit/search tools and wires them to a backend
- **Subagent middleware**, adds the `task` tool and manages spawning and isolating subagent context
- **Summarization/compaction middleware**, automatically compresses long conversation history and offloads large tool outputs to disk instead of letting them sit in the message list forever
- **Human-in-the-loop middleware**, lets you require approval before specific tool calls execute, built on the same `interrupt()` mechanism from Chapter 10

Because it's middleware, you can swap or override any piece. Don't want the default filesystem backend, pass your own `FilesystemMiddleware` instance. Don't want a subagent to inherit filesystem access, give that subagent its own middleware config, subagents declared this way don't automatically inherit the parent's middleware.

This is also the direct answer to why deep agents fix the "context window overflow" problem from Chapter 27: the summarization/compaction middleware is doing this automatically, in the background, without you writing any manual truncation logic.

---

## Chapter 31: Files a Deep Agent Creates to Preserve Context

Beyond whatever files the agent writes for your specific task (a `summary.md`, notes per source, and so on), the harness uses a few conventions to keep context under control and persist knowledge across sessions:

- **`AGENTS.md`**, a file the memory middleware reads on startup to load persistent context and instructions across sessions, the same file survives a restart, which is how a deep agent "remembers" project-level context without that context ever occupying space in the live message history.
- **Offloaded tool outputs**, when a tool call returns something large (a long document, a big search result), the context management middleware can write the full output to a file and leave only a short reference or summary in the message list, keeping the actual context window lean while the full content stays retrievable if needed.
- **Compaction summaries**, when a conversation grows past a threshold, the summarization middleware compresses older turns into a summary, again keeping the working context small without discarding the information outright.

The pattern across all three is the same: don't let information disappear just because it's not in the live context anymore, put it somewhere durable and let the agent go fetch it again if it turns out to matter.

---

## Chapter 32: When to Reach for What

Three layers, in increasing order of how much you're building yourself:

- **LangGraph** (Part II), when the agent loop itself isn't the right shape and you need custom routing, custom state, or a workflow that isn't a simple reason-act loop at all.
- **`create_agent`** (plain ReAct, Part I question you asked earlier), when you want a lighter agent without the bundled planning/filesystem/subagent machinery, a single tool-calling loop is genuinely enough for the task.
- **`create_deep_agent`**, when the task is long-running, multi-step, or would otherwise blow past context limits, the kind of thing that benefits from planning, delegation, and persistent files. This is the architecture behind tools like Claude Code and other long-horizon coding/research agents.

They compose rather than compete: any compiled LangGraph graph can be passed into a deep agent as a subagent, so custom orchestration you build in Part II style slots directly into the Part VII harness when you need both.

---

Guardrails


## Summary: Deep Agents

| Concept                                                                         | Covered |
| ---------------------------------------------------------------------------------| ---------|
| Shallow agent / ReAct loop, and why it breaks on long tasks                     | Yes     |
| `write_todos`, the planning tool                                                | Yes     |
| Subagents and the `task` tool, isolated context + delegation                    | Yes     |
| Virtual filesystem, shared persistent memory across agents                      | Yes     |
| System prompt's role in a deep agent                                            | Yes     |
| Default middleware stack (planning, filesystem, subagents, summarization, HITL) | Yes     |
| `AGENTS.md` and context-preserving files                                        | Yes     |
| When to use LangGraph vs `create_agent` vs `create_deep_agent`                  | Yes     |

## Summary: LangGraph Core

| Concept | Covered | Project |
|---|---|---|
| Gen AI vs Agentic AI | Yes | Theory |
| What is LangGraph, why not just LangChain | Yes | Theory |
| Sequential workflow | Yes | `script_writter` |
| Parallel workflow (fan-out/fan-in) | Yes | `AI_content_Moderator` |
| Conditional workflow | Yes | `chat_bot` |
| Reducers (custom + `add_messages`) | Yes | Both |
| `TypedDict` state, `Annotated` | Yes | All projects |
| Router functions, conditional edges | Yes | `chat_bot` |
| `ToolNode`, `@tool`, `bind_tools`, `tools_condition` | Yes | `chat_bot` |
| `MemorySaver`, thread ID | Yes | `chat_bot` |
| `stream` / `astream` / `stream_mode` | Yes | `chat_bot` |
| `interrupt()` / `Command` (HITL) | Yes | Concept, not yet in a project |
| LangSmith | No | Not yet |
| Persistent checkpointers | No | Not yet |
| Structured output | No | Not yet |
| Multi-agent systems | No | Not yet |
| Subgraphs | No | Not yet |

## Summary: MCP

| Concept | Covered |
|---|---|
| Host / Client / Server roles | Yes |
| Tools / Resources / Prompts | Yes |
| STDIO, Streamable HTTP, deprecated SSE | Yes |
| `langchain-mcp-adapters` | Yes |
| 2026 spec changes (stateless core, MCP Apps, Tasks) | Yes |

## Summary: RAG

| Concept | Covered |
|---|---|
| Full pipeline (load, split, embed, store, retrieve, augment) | Yes |
| Document structure and metadata filtering | Yes |
| Loaders | Yes |
| Chunking types | Yes |
| Embeddings | Yes |
| Vector DBs | Yes |
| Retrieval (+ MMR, hybrid, reranking) | Yes |
| Augmentation | Yes |
| Agentic RAG (RAG as a tool) | Yes |
| Traditional RAG, pros and cons | Yes |
| Vectorless RAG (tree-based, PageIndex) | Yes |
| RAG evaluation | No |

---

You've covered a genuinely wide range here for where you are in this. The foundation is solid, keep building on it.