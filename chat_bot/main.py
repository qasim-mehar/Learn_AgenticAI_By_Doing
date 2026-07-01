from rich import print
from typing import Annotated, TypedDict
from typing import Annotated
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langgraph.graph import add_messages
from langchain_mistralai import ChatMistralAI
from langchain_tavily.tavily_search import TavilySearch
from dotenv import load_dotenv

load_dotenv()


class State(TypedDict):
    """
    State schema for a LangGraph chatbot application.

    This TypedDict defines the structure of data that flows through the agent graph.
    Each message from the user and assistant passes through this state, with the
    messages list being intelligently merged rather than replaced.

    Attributes
    ----------
    messages : Annotated[list, add_messages]
        A list of message objects representing the conversation history.

        The `Annotated` wrapper with `add_messages` reducer tells LangGraph:
        - Type hint: this is a list
        - Metadata: use the `add_messages` reducer function
        - Behavior: when updating state, merge new messages instead of replacing

        Example flow:
        - Initial state: messages = []
        - Node 1 adds: [HumanMessage("Hi")]
        - State becomes: [HumanMessage("Hi")]
        - Node 2 adds: [AIMessage("Hello")]
        - State becomes: [HumanMessage("Hi"), AIMessage("Hello")]

        WITHOUT Annotated + add_messages, Node 2 would REPLACE the list entirely,
        losing the human message. The reducer ensures conversation history accumulates.

    Notes
    -----
    This is a minimal state for a basic chatbot. You can extend it:

        class State(TypedDict):
            messages: Annotated[list, add_messages]
            user_id: str              # Track which user
            research_results: str     # Store tool outputs
            agent_decision: str       # Store what agent decides next

    See Also
    --------
    langgraph.graph.add_messages : Reducer that merges message lists intelligently
    LangGraph State management: https://langchain-ai.github.io/langgraph/concepts/
    """

    messages: Annotated[list, add_messages]


llm = ChatMistralAI(model="mistral-medium-3-5")


# NODE FUNCTIONALITY
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


# PASS STATE IN STATEGRAPH SO WE CAN ACCESS THIS STATE ANYWHERE IN THE GRAPH
graph_builder = StateGraph(State)

graph_builder.add_node("chatbotNode", chatbot)
graph_builder.add_edge(START, "chatbotNode")
graph_builder.add_edge("chatbotNode", END)

graph = graph_builder.compile()

# graph.invoke({"messages": "hi"})
