import streamlit as st
from openai import OpenAI

# Set page title
st.title("Chat Demo")

# Sidebar for model selection
with st.sidebar:
    model = st.radio(
        "Select an LLM:",
        ['llava', 'gemma2', 'phi3', 'llama3', 'embed-mistral', 'mixtral', 'gorilla', 'groq-tools'],
        index=0
    )
    st.session_state["model"] = model

# Load API key from environment or secrets
import os
api_key = os.getenv("LITELLM_KEY")
if api_key is None:
    api_key = st.secrets["LITELLM_KEY"]

# Initialize chat messages session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Initialize OpenAI client
client = OpenAI(
    api_key=api_key,
    base_url="https://llm.nrp-nautilus.io"
)

# Clear session state and reset chat history
if st.button("Clear History"):
    st.session_state.messages = []
    st.experimental_rerun()

# Chat input element for user prompts
if prompt := st.chat_input("Ask me something!"):
    # Store user message in session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # AI response processing
    with st.chat_message("assistant"):
        try:
            stream = client.chat.completions.create(
                model=st.session_state["model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )
            response = "".join([chunk["choices"][0]["delta"]["content"] for chunk in stream])
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"Error: {e}")

# Feedback section
with st.expander("Provide Feedback"):
    feedback = st.radio(
        "Was this response helpful?",
        options=["Yes", "No", "Partially"],
        index=2
    )
    if feedback:
        st.write(f"Thank you for your feedback: {feedback}")
