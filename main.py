from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from transformers import pipeline

# ----------------------------
# Load TinyLlama
# ----------------------------
piped_llm = pipeline(
    task="text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device=-1
)

llm = HuggingFacePipeline(
    pipeline=piped_llm,
    model_kwargs={
        "temperature": 0.7,
        "max_new_tokens": 200,
        "return_full_text": False
    }
)

model = ChatHuggingFace(llm=llm)

# ----------------------------
# Prompt Template with Placeholder
# ----------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# Store chat history
chat_history = []

# ----------------------------
# Chat Loop
# ----------------------------
while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    formatted_prompt = prompt.invoke({
        "input": user_input,
        "chat_history": chat_history
    })

    response = model.invoke(formatted_prompt)

    print("AI:", response.content.strip())

    # Update memory
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response.content))