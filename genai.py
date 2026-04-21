from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import pipeline
import streamlit as st


st.header("Research Tool")

# -----------------------------
# Create HF text generation pipeline
# -----------------------------
piped_llm = pipeline(
    task="text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device=-1  # CPU (change to 0 if you have GPU)
)

# -----------------------------
# Wrap in LangChain
# -----------------------------
llm = HuggingFacePipeline(
    pipeline=piped_llm,
    model_kwargs={
        "temperature": 0.5,
        "max_new_tokens": 150
    }
)

model = ChatHuggingFace(llm=llm)

# -----------------------------
# Streamlit UI
# -----------------------------
user_input = st.text_input("Enter your question")

if st.button("Summarize"):
    if user_input.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Summarizing..."):
            result = model.invoke(user_input)
            st.write(result.content)