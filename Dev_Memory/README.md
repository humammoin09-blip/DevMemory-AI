<div align="center">

# DevMemory-AI

### *State-Aware, Context-Intelligent Autonomous Developer Assistant*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white&style=for-the-badge)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-ff69b4.svg?logo=langchain&logoColor=white&style=for-the-badge)](https://langchain-ai.github.io/langgraph/)
[![LangChain](https://img.shields.io/badge/Framework-LangChain-1C3C3C.svg?logo=langchain&logoColor=white&style=for-the-badge)](https://www.langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)
[![Maintenance](https://img.shields.io/badge/Maintenance-Active-emerald.svg?style=for-the-badge)](#)

<p align="center">
  <b>DevMemory-AI</b> bridges the critical gap in modern LLM developer tooling: persistent cognitive state. Engineered with dual-layer memory hierarchies (Short-Term Conversational Checkpointing + Hierarchical Long-Term Namespaced Storage) over a cyclic ReAct graph execution loop.
</p>

---

</div>

## Table of Contents
- [1. Overview & Problem Statement](#1-overview--problem-statement)
- [2. System Architecture & Flow](#2-system-architecture--flow)
- [3. Core Technical Highlights](#3-core-technical-highlights)
  - [Dual-Layer Cognitive Memory Architecture](#dual-layer-cognitive-memory-architecture)
  - [Dynamic Context Injection & Agent Personalization](#dynamic-context-injection--agent-personalization)
  - [Cyclic ReAct Execution Engine](#cyclic-react-execution-engine)
- [4. Custom Developer Tools](#4-custom-developer-tools)
- [5. Technology Stack](#5-technology-stack)
- [6. Repository Structure](#6-repository-structure)
- [7. Getting Started & Prerequisites](#7-getting-started--prerequisites)
- [8. Installation & Setup](#8-installation--setup)
- [9. Environment Configuration](#9-environment-configuration)
- [10. Usage & Code Walkthrough](#10-usage--code-walkthrough)
- [11. Production Roadmap](#11-production-roadmap)
- [12. License & Author](#12-license--author)

---

## 1. Overview & Problem Statement

Standard Large Language Model (LLM) agents operate as **stateless reasoning engines**. While chat sessions can emulate continuity by replaying token buffers within an active window, they suffer from fundamental architectural flaws:

1. **Context Window Degradation & Token Bleed:** Appending unbounded historical dialogue explodes inference costs and degrades retrieval fidelity (the "needle-in-a-haystack" degradation).
2. **Session Isolation:** Once a session concludes or a thread terminates, developer profile knowledge—such as preferred tech stacks, test conventions, architecture patterns, and stylistic nuances—is irrevocably lost.
3. **Unstructured Recall:** Ad-hoc retrieval fails to distinguish ephemeral thread-level state from persistent cross-session knowledge.

**DevMemory-AI** solves this through a formalized, graph-based architecture built on **LangGraph**. By decoupling **Short-Term Memory (STM)** via transactional thread checkpointing from **Long-Term Memory (LTM)** via hierarchical key-value namespaces, DevMemory-AI maintains rich developer profiles across disparate sessions while executing autonomous tool invocations in a robust ReAct loop.

---

## 2. System Architecture & Flow

DevMemory-AI models the agent lifecycle as an explicit state machine utilizing LangGraph's `StateGraph`.

```
                    ┌─────────────────────────┐
                    │          START          │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │    Profile & Context    │
                    │     LTM Extraction      │
                    │ (Namespaces via Store)  │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
       ┌───────────►│        chat_node        │
       │            │  (System Prompt + LTM)  │
       │            └────────────┬────────────┘
       │                         │
       │                [tools_condition]
       │                         │
       │         ┌───────────────┴───────────────┐
       │         │ (Tool Calls Detected)         │ (No Tool Calls)
       │         ▼                               ▼
       │   ┌───────────┐                   ┌───────────┐
       └───┤ ToolNode  │                   │    END    │
           │ Execution │                   │ (Save STM)│
           └───────────┘                   └───────────┘
```

### Execution Lifecycle:
1. **Invocation:** The graph receives an input payload parameterized by `thread_id` (STM scope) and `user_id` (LTM scope).
2. **Context Synthesis:** In `chat_node`, the runtime reads the LTM store at `("users", user_id, "details")`, resolves stored developer preferences, and injects them into the foundational system instructions.
3. **Model Inference:** The LLM generates a response or emits structured tool-call vectors.
4. **Conditional Routing (`tools_condition`):**
   - If tool calls exist: State routes to `ToolNode`, runs the tools synchronously/asynchronously, commits outputs as `ToolMessage` instances, and loops back to `chat_node`.
   - If no tool calls exist: Execution transitions to `END`, and the `MemorySaver` atomically checkpoints the state against the `thread_id`.

---

## 3. Core Technical Highlights

### Dual-Layer Cognitive Memory Architecture

DevMemory-AI strictly bifurcates state into ephemeral thread buffers and durable namespace stores:

| Attribute | Short-Term Memory (STM) | Long-Term Memory (LTM) |
| :--- | :--- | :--- |
| **Backing Engine** | `MemorySaver` (Checkpointer) | `InMemoryStore` / `PostgresStore` |
| **Addressing Scheme** | `configurable: {"thread_id": "<uuid>"}` | Hierarchical tuple namespaces: `("users", user_id, "details")` |
| **Lifespan** | Ephemeral or thread-bound | Durable across independent executions |
| **Primary Role** | Dialogue continuity, ReAct intermediate outputs | Architectural preferences, stylistic rules, tech stacks |
| **Scaling Vector** | Sliding window / message summarization | Indexed semantic & key-value queries |

### Dynamic Context Injection & Agent Personalization
Unlike naive vector augmentations, DevMemory-AI treats developer configurations deterministically:
- Developers register preferences (e.g., `"Primary Stack: FastAPI, TypeScript; Strict typing; TDD with Pytest"`).
- These profiles are stored under structured namespaces.
- The `chat_node` intercepts incoming state and prepends an adaptive system directive containing the active profile before prompt resolution, eliminating manual context re-prompting.

### Cyclic ReAct Execution Engine
Constructed on LangGraph's cyclic paradigm:
- Native schema validation via **Pydantic**.
- Deterministic routing edges powered by `@langchain/core` / `tools_condition`.
- Zero infinite-loop traps through graph recursion limits and validated execution traces.

---

## 4. Custom Developer Tools

DevMemory-AI includes production-hardened tools purpose-built for engineering workflows:

### 1. Robust Engineering Calculator (`calculator`)
- **Purpose:** Handles complex computational logic, algorithmic complexity assessments, and memory-footprint calculations.
- **Safety:** Sandboxed evaluation supporting arithmetic and advanced mathematical functions via `math` primitives, avoiding insecure global `eval`.

### 2. Deep Code & Text Analyzer (`analyze_text_code`)
- **Purpose:** Real-time lexical analysis of snippets, diffs, documentation, and query strings.
- **Metrics Tracked:**
  - Raw character count (including & excluding whitespace).
  - Total word tokens.
  - Estimated sentence count / line density.
  - Code block presence and syntax-level heuristic insights.

---

## 5. Technology Stack

- **Core Runtime & Language:** Python `3.10+`
- **Graph Orchestration & State Engines:** [LangGraph](https://github.com/langchain-ai/langgraph) (`StateGraph`, `MemorySaver`, `InMemoryStore`, `ToolNode`)
- **LLM Abstraction Layer:** [LangChain Core](https://github.com/langchain-ai/langchain) (`@langchain/core`)
- **Supported Providers:** 
  - Google Gemini API via `langchain-google-genai`
  - OpenAI API via `langchain-openai`
- **Data Modeling & Validation:** [Pydantic v2](https://docs.pydantic.dev/)
- **Configuration & Environment Management:** `python-dotenv`

---

## 6. Repository Structure

```
DevMemory-AI/
├── src/
│   ├── __init__.py
│   ├── agent.py               # StateGraph definition, chat_node, and compilation
│   ├── state.py               # AgentState schema (TypedDict / Pydantic models)
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── checkpointer.py    # MemorySaver (STM) configurations
│   │   └── store.py           # InMemoryStore / PostgresStore (LTM) management
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── calculator.py      # Math & algorithmic calculation tool
│   │   └── text_analyzer.py   # Code / Text lexical inspection tool
│   └── utils/
│       ├── __init__.py
│       └── logger.py          # Structured logging and output formatting
├── tests/
│   ├── __init__.py
│   ├── test_memory.py         # STM checkpointing & LTM persistence tests
│   └── test_tools.py          # Unit tests for Calculator & Text Analyzer
├── .env.example               # Template for required environment variables
├── .gitignore                 # Standard Python, virtualenv, and cache ignores
├── LICENSE                    # MIT License
├── pyproject.toml             # Poetry / modern packaging specification
├── requirements.txt           # Pinned production dependencies
└── README.md                  # System architecture and documentation
```

---

## 7. Getting Started & Prerequisites

### Prerequisites
- **Python:** Version `3.10` or higher installed. Check with:
  ```bash
  python --version
  ```
- **API Keys:** Either an active **Google Gemini API Key** or an **OpenAI API Key**.
- **Virtual Environment Tool:** `venv` or `conda`.

---

## 8. Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/DevMemory-AI.git
cd DevMemory-AI
```

### 2. Initialize Virtual Environment
```bash
# On Linux / macOS
python3 -m venv venv
source venv/bin/activate

# On Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 9. Environment Configuration

Create a `.env` file from the provided sample:

```bash
cp .env.example .env
```

Populate the configuration values according to your target model provider:

```ini
# ==============================================================================
# Model Provider Credentials (Configure at least one)
# ==============================================================================
GEMINI_API_KEY="your-gemini-api-key-here"
OPENAI_API_KEY="your-openai-api-key-here"

# Model Selection
DEFAULT_MODEL_PROVIDER="gemini" # Options: "gemini" | "openai"
DEFAULT_MODEL_NAME="gemini-1.5-pro-latest"

# ==============================================================================
# Runtime & Persistence Configurations
# ==============================================================================
DEBUG=False
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=""

# (Optional) Future PostgreSQL Store Extension
# POSTGRES_CONNECTION_STRING="postgresql://postgres:password@localhost:5432/devmemory"
```

---

## 10. Usage & Code Walkthrough

Below is a complete, minimal walkthrough illustrating how LTM and STM interact during graph execution.

```python
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from src.state import AgentState
from src.tools.calculator import calculator
from src.tools.text_analyzer import analyze_text_code
from src.agent import chat_node

load_dotenv()

# 1. Initialize Short-Term Memory Checkpointer and Long-Term Memory Store
checkpointer = MemorySaver()
ltm_store = InMemoryStore()

# 2. Seed Developer Profile into Long-Term Memory (LTM)
user_id = "dev_lead_01"
ltm_store.put(
    namespace=("users", user_id, "details"),
    key="profile",
    value={
        "name": "Alex",
        "primary_stack": "Python, FastAPI, Rust",
        "coding_style": "Type-safe, Clean Architecture, Docstrings required",
        "test_preference": "Pytest with high coverage"
    }
)

# 3. Construct and Compile the StateGraph
tools = [calculator, analyze_text_code]
workflow = StateGraph(AgentState)

workflow.add_node("chat_node", chat_node)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "chat_node")
workflow.add_conditional_edges("chat_node", tools_condition)
workflow.add_edge("tools", "chat_node")

# Compile with both checkpointer and store
app = workflow.compile(checkpointer=checkpointer, store=ltm_store)

# 4. Invoke the Agent
config = {
    "configurable": {
        "thread_id": "session-thread-alpha-101", # STM Scope
        "user_id": user_id                       # LTM Scope
    }
}

input_payload = {
    "messages": [
        HumanMessage(content="Recommend an API structure for our new payment microservice.")
    ]
}

response = app.invoke(input_payload, config=config)

# Output response tailored specifically to Alex's stored profile
print(response["messages"][-1].content)
```

---

## 11. Production Roadmap

- [x] Dual-tier state persistence (STM checkpointer + LTM namespaced store).
- [x] Cyclic ReAct loop with automated conditional tool invocation.
- [x] Dynamic context injection based on caller identity metadata.
- [ ] **Enterprise LTM Scaling:** Transition from `InMemoryStore` to distributed `PostgresStore` with Docker Compose integration.
- [ ] **Semantic Memory Search:** Vector-indexed semantic queries across past project decisions and design docs via `pgvector`.
- [ ] **Interactive Developer UI:** Full-featured Web GUI implemented via [Streamlit](https://streamlit.io/) or Next.js.
- [ ] **Multi-Agent Orchestration:** Specialized subgraphs for automated Code Review, Security Auditing, and Documentation generation.

---

## 12. License & Author

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for full details.

Developed and maintained with precision by **Senior AI & Systems Engineers**. Contributions, bug reports, and architectural RFCs are warmly welcomed via pull requests.
