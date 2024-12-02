import streamlit as st
from langchain_community.document_loaders import PyPDFLoader

# Use API key from environment or Streamlit secrets
import os
api_key = os.getenv("LITELLM_KEY")
if api_key is None:
    api_key = st.secrets["LITELLM_KEY"]

# Set the page title
st.title("RAG Demo - Seattle Geology")

'''
Provide a URL to a PDF document you want to ask questions about.
Once the document has been uploaded and parsed, ask your questions in the chat dialog that will appear below.
'''

# PDF URL input
url = st.text_input(
    "PDF URL",
    "https://your.kingcounty.gov/dnrp/library/water-and-land/science/seminars/November-2004/Mapping-Geology-of-Greater-Seattle-Area-Infiltration-Peat-Bogs-and-Volcanic-Ash.pdf"
)

# Function to load PDF content
@st.cache_data
def pdf_loader(url):
    loader = PyPDFLoader(url)
    return loader.load()

# Load the PDF content
docs = pdf_loader(url)

# Set up the language model
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    model="llama3",  # Use other models like "gorilla" or "phi3" as needed
    api_key=api_key,
    base_url="https://llm.nrp-nautilus.io",
    temperature=0
)

# Set up the embedding model
from langchain_openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings(
    model="embed-mistral",
    api_key=api_key,
    base_url="https://llm.nrp-nautilus.io"
)

# Build a retriever using the embedding model
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Split documents into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Store and retrieve document embeddings
vectorstore = InMemoryVectorStore.from_documents(documents=splits, embedding=embedding)
retriever = vectorstore.as_retriever()

# Create a retrieval-based question-answering chain
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Define system prompt for concise question answering
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Set up the retrieval-based QA chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Streamlit chat interface
if question := st.chat_input("Ask a question about the Seattle geology document:"):
    # Display the user's question
    with st.chat_message("user"):
        st.markdown(question)

    # Retrieve and display the assistant's response
    with st.chat_message("assistant"):
        try:
            results = rag_chain.invoke({"input": question})
            st.markdown(results['answer'])

            # Expandable section for matched context
            with st.expander("See context matched"):
                st.write(results['context'][0].page_content)
                st.write(results['context'][0].metadata)
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Note: For multi-turn interactions, LangChain memory-based QA can be implemented.