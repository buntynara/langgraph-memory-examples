import os
import operator

from dotenv import load_dotenv
from typing_extensions import Annotated, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.postgres import PostgresSaver

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

class ChatState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int

def chat_node(state: ChatState):
    print("🤖 Agent is responding...")

    messages = [
        SystemMessage(
            content=(
                "You are a helpful AI assistant. "
                "Use conversation memory properly and answer based on previous messages."
            )
        ),
        *state["messages"],
    ]

    response = llm.invoke(messages)

    # update the state with the new message and increment the llm_calls counter
    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


def build_graph(checkpointer):
    """
    Build a state graph for a simple chat application with given checkpointer for memory management.
    """
    builder = StateGraph(ChatState)

    builder.add_node("chat", chat_node)
    builder.add_edge(START, "chat")
    builder.add_edge("chat", END)

    return builder.compile(checkpointer=checkpointer)


def invoke_graph(message: str, graph, config):
    """invoke the graph with a message and configuration"""
    return graph.invoke(
        {
            "messages": [
                HumanMessage(content=message)
            ],
            "llm_calls": 0
        },
        config=config
    )

def main():
    # long term memory is implemented using a PostgresSaver, which persists data across sessions.
    with PostgresSaver.from_conn_string(DATABASE_URL) as checkpointer:
        checkpointer.setup()

        graph = build_graph(checkpointer)
        config = {"configurable": {"thread_id": "user_session_1234"}}
        
        # run this first to set the context for the conversation
        result = invoke_graph("My name is Bunty. I work as a GenAI Engineer, Sr. Technical Lead.", graph, config)
        print(result["messages"][-1].content) # Output: Nice to meet you, Bunty...


        # run this second time to see if the model remembers the context from the previous message
        # result = invoke_graph("what is my name?", graph, config)
        # print(result["messages"][-1].content) # Output: We've already met, Bunty. You're a GenAI Engineer...



if __name__ == "__main__":
    main()





