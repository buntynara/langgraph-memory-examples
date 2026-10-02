import operator

from dotenv import load_dotenv
from typing_extensions import Annotated, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
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
    # short term memory is implemented using an in-memory saver, which does not persist data across sessions.
    checkpointer = InMemorySaver()
    
    graph = build_graph(checkpointer)
    config = {"configurable": {"thread_id": "user_session_1234"}}

    # run this first to set the context for the conversation
    # result = invoke_graph("My name is Bunty. I work as a GenAI Engineer, Sr. Technical Lead.", graph, config)
    # print(result["messages"][-1].content) # Output: Nice to meet you, Bunty...


    # run this second time to see if the model remembers the context from the previous message
    result = invoke_graph("what is my name?", graph, config)
    print(result["messages"][-1].content) # Output: I don't have any information about your name...



if __name__ == "__main__":
    main()