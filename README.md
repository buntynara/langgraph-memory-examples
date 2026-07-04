# LangGraph Memory

A hands-on Python project demonstrating conversational memory and state persistence in LangGraph.

The project compares in-memory checkpointing with PostgreSQL-backed persistent checkpointing to show how conversation state can be maintained across interactions and application restarts.

## Features

* Stateful conversations using LangGraph
* Custom graph state with message history and LLM call tracking
* Short-term memory using `InMemorySaver`
* Persistent conversation state using PostgreSQL
* Thread-based conversation isolation
* Groq LLM integration
* Environment-based configuration

## Project Structure

```text
langgraph-memory/
├── short_term_memory.py
├── long_term_memory.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Memory Examples

### In-Memory Checkpointing

`short_term_memory.py` uses LangGraph's `InMemorySaver` to maintain conversation state within the application process.

Conversation history is associated with a `thread_id`, allowing the graph to retain context across multiple invocations for the same thread.

The stored state is lost when the application process terminates.

### PostgreSQL-Backed Checkpointing

`long_term_memory.py` uses `PostgresSaver` to persist graph checkpoints in PostgreSQL.

Conversation state can be restored across application restarts when the same `thread_id` is used.

This demonstrates durable LangGraph state persistence using an external database.

## Tech Stack

* Python
* LangGraph
* LangChain Core
* Groq
* PostgreSQL
* Psycopg 3

## Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd langgraph-memory
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy the example environment file:

```bash
cp .env.example .env
```

Update `.env` with your configuration:

```dotenv
GROQ_API_KEY=your_groq_api_key
DATABASE_URL=postgresql://username:password@localhost:5432/database_name
```

## Run the Examples

### In-memory checkpointing

```bash
python short_term_memory.py
```

### PostgreSQL-backed checkpointing

Ensure PostgreSQL is running and the configured database exists.

```bash
python long_term_memory.py
```

On the first execution, the PostgreSQL checkpointer initializes the required checkpoint tables.

## Key Concept

LangGraph checkpointing persists graph state at thread boundaries.

A configurable `thread_id` identifies a conversation thread:

```python
config = {
    "configurable": {
        "thread_id": "user_session_1234",
    }
}
```

Invocations using the same `thread_id` continue with the previously checkpointed conversation state.

Using a different `thread_id` creates an independent conversation context.

## Learning Objective

This project explores how LangGraph manages stateful LLM workflows and how checkpoint storage can be changed from process-local memory to durable PostgreSQL persistence.

It provides a foundation for building stateful AI assistants and agentic applications that require persistent conversation context.
