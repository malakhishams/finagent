# FinAgent — Conversational Financial AI Assistant

A LangChain-powered financial assistant built in two complementary scripts:
a **memory-based conversational agent** and a **ReAct RAG agent** over a real financial report.

Built as part of Sprints AI\ML scholarship curriculum.

---

##  Project Structure

```
finagent/
├── memory_tool_chain.py   # Conversational agent with memory + FAISS retrieval
├── react_finagent.py      # ReAct agent with RAG over a financial report
├── data/
│   └── report.txt         # Nexora Technologies FY2024 financial report
├── settings.py            # API key and model config (not committed) 
└── .gitignore
```

---

## Two Agents, Two Approaches

### 1. `memory_tool_chain.py` — Conversational Agent with Memory

A multi-turn chat agent that remembers everything said earlier in the conversation and can look up financial term definitions on demand.

**How it works:**

```
User Input
    │
    ▼
InMemoryChatMessageHistory  ──►  Full conversation injected into prompt
    │
    ▼
Gemini LLM (with tool access)
    │
    ├── If financial term query ──► FAISS retriever tool ──► definition
    └── If factual recall ──────► answers from memory
    │
    ▼
Response + memory updated
```

**Key design choices:**
- `InMemoryChatMessageHistory` instead of deprecated `ConversationBufferMemory` — keeps the full conversation verbatim
- WindowMemory was rejected: loses early context by Turn 3
- SummaryMemory was rejected: compresses history, losing exact figures like `$10M`
- FAISS vector store holds 3 financial facts (EBITDA, Revenue, Net Profit) — retrieved via `create_retriever_tool`

**Example interaction:**
```
Turn 1 → User: "The company's Q1 revenue was $10M."
Turn 2 → User: "What was the Q1 revenue?"  →  Agent recalls $10M from memory
Turn 3 → User: "What is EBITDA?"  →  Agent calls FAISS retriever tool
```

---

### 2. `react_finagent.py` — ReAct Agent with RAG Pipeline

A reasoning agent that reads a real financial report (`report.txt`), chunks and embeds it into a Chroma vector store, then answers multi-step questions using a Thought → Action → Observation → Answer loop.

**RAG pipeline:**

```
report.txt
    │
    ▼
TextLoader → RecursiveCharacterTextSplitter (chunk=500, overlap=50)
    │
    ▼
HuggingFace Embeddings (all-MiniLM-L6-v2)
    │
    ▼
Chroma Vector Store
    │
    ▼
Retriever Tool (top-3 chunks)
    │
    ▼
ReAct Agent (Gemini) → Thought → Action → Observation → Final Answer
```

**Example query:**
```
"What was the Q4 2024 profit margin?
 What was the Q3 2024 total revenue?
 Compare the two figures."
```
The agent autonomously decides how many times to call the retrieval tool before composing a final answer.

---

## Tech Stack

| Component | Technology |
|---|---|
| Framework | LangChain 1.2.18, LangGraph |
| LLM | Google Gemini (`gemini-2.5-flash`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Stores | FAISS (`memory_tool_chain.py`), Chroma (`react_finagent.py`) |
| Memory | `InMemoryChatMessageHistory` (LangChain Core) |
| Agent | `create_react_agent` (LangGraph prebuilt) |
| Language | Python 3.10+ |

---

## Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/malakhishams/finagent.git
cd finagent
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install langchain langchain-core langchain-community langgraph
pip install langchain-google-genai langchain-huggingface
pip install faiss-cpu chromadb sentence-transformers
```

### 4. Configure API credentials

Create a `settings.py` file in the project root:
```python
GEMINI_API_KEY = "your-gemini-api-key"
GEMINI_MODEL = "gemini-2.5-flash"
```

### 5. Run the agents

**Conversational memory agent:**
```bash
python memory_tool_chain.py
```

**ReAct RAG agent:**
```bash
python react_finagent.py
```

---

## ⚠️ LangChain 1.2.18 Compatibility Notes ⚠️

Several commonly documented patterns are deprecated in this version:

| Deprecated | Used Instead |
|---|---|
| `ConversationBufferMemory` | `InMemoryChatMessageHistory` (langchain_core) |
| `ConversationChain` | Custom LCEL chain with `MessagesPlaceholder` |
| `langchain.agents.create_react_agent` | `langgraph.prebuilt.create_react_agent` |

---

## License

Developed as part of sprints AI\ML  scholarship program. Feel free to use or adapt with attribution.