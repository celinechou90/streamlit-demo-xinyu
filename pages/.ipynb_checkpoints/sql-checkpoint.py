import streamlit as st
import os
import sqlalchemy
import ibis
from ibis import _
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import create_sql_query_chain

# Set the page title
st.title("SQL Demo - Seattle")

# Load the API key from environment variables or Streamlit secrets
api_key = os.getenv("LITELLM_KEY") or st.secrets.get("LITELLM_KEY")
if not api_key:
    st.error("API key is missing. Please add it to Streamlit secrets or set as an environment variable.")

# Parquet file input field
parquet = st.text_input("Parquet file:", placeholder="Enter Parquet file URL or leave blank")

# Create a DuckDB engine connection
try:
    eng = sqlalchemy.create_engine("duckdb:///:memory:")  # DuckDB in-memory database
    ibis_con = ibis.duckdb.connect()  # Establishing connection with ibis
    tbl = None

    # Load Parquet file into DuckDB using Ibis if provided
    if parquet:
        tbl = ibis_con.read_parquet(parquet, "mydata")
        st.write(f"Successfully loaded Parquet file: {parquet}")
    else:
        st.info("No Parquet file provided. Please input one to query data.")
except Exception as e:
    st.error(f"Failed to set up DuckDB or load Parquet file: {e}")

# Connect LangChain to the DuckDB engine
try:
    db = SQLDatabase(eng, view_support=True)
except Exception as e:
    st.error(f"Failed to connect LangChain to the DuckDB engine: {e}")

# Define the system prompt template
template = '''
You are a {dialect} expert. Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer to the input question.
Always return all columns from a query (select *) unless otherwise instructed.
Wrap each column name in double quotes (") to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below. 
Be careful to not query for columns that do not exist. 
Also, pay attention to which column is in which table.
Pay attention to use today() function to get the current date, if the question involves "today".
Respond with only the SQL query to run. Do not repeat the question or explanation. Just the raw SQL query.
Only use the following tables:
{table_info}
Question: {input}    
'''

prompt_template = PromptTemplate.from_template(template, partial_variables={"dialect": "duckdb", "top_k": 10})

# Set up the LLM
try:
    llm = ChatOpenAI(
        model="gorilla",  # Options: llama3, gorilla, groq-tools, etc.
        temperature=0,
        api_key=api_key,
        base_url="https://llm.nrp-nautilus.io"
    )
except Exception as e:
    st.error(f"Failed to initialize the LLM: {e}")

# Create the SQL query chain
try:
    chain = create_sql_query_chain(llm, db, prompt_template)
except Exception as e:
    st.error(f"Failed to create the SQL query chain: {e}")

# Chat input for SQL query questions
if prompt := st.chat_input("Ask a question (e.g., What is the mean NDVI by grade?)"):
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and execute SQL query
    with st.chat_message("assistant"):
        try:
            response = chain.invoke({"question": prompt})
            st.markdown(f"Generated SQL Query:\n```sql\n{response}\n```")

            # Execute the generated query if Parquet file was loaded
            if tbl:
                df = ibis_con.sql(response).execute()
                st.write(df)
            else:
                st.warning("No Parquet file was loaded. Unable to execute the query.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
