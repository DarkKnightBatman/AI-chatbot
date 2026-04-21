from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from transformers import pipeline
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# LLM Setup
pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=150
)

llm_pipeline = HuggingFacePipeline(pipeline=pipe)
model = ChatHuggingFace(llm=llm_pipeline)

# Graph State
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Node
def chat_node(state: ChatState):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}

# Memory
checkpointer = MemorySaver()

# Graph
graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

# Compile chatbot
chatbot = graph.compile(checkpointer=checkpointer)