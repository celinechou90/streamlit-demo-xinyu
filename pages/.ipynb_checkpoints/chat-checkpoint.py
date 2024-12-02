import streamlit as st
from openai import OpenAI

st.title("Chat Demo")

with st.sidebar:
    model = st.radio("Select an LLM:", ['llava', 'gemma2', 'phi3', 'llama3', 'embed-mistral', 'mixtral', 'gorilla', 'groq-tools'])
    st.session_state["model"] =  model

## dockerized streamlit app wants to read from os.getenv(), otherwise use st.secrets
import os
api_key = os.getenv("LITELLM_KEY")
if api_key is None:
    api_key = st.secrets["LITELLM_KEY"]
        

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

client = OpenAI(
    api_key = api_key, 
    base_url = "https://llm.nrp-nautilus.io"
)

# Button to clear session state
if st.button('Clear History'):
    st.session_state.clear()

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})
