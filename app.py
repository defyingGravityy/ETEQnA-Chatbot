import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
groq_api_key=os.getenv("GROQ_API_KEY")
llm= ChatGroq(groq_api_key=groq_api_key, model="Llama3-8b-8192")


#langsmith trackes
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="TRUE"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")


prompt= ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant.Answer the question to the best of your ability."),
        ("user","Question: {question}")
    ]
)

def generate_answer(question,api_key,llm,temperature=0.7, max_tokens=1000) -> str:
    llm_groq_api_key = groq_api_key
    llm = ChatGroq(groq_api_key=llm_groq_api_key, model="Llama3-8b-8192", temperature=temperature, max_tokens=max_tokens)
    output_parsers=StrOutputParser()
    chain= prompt | llm | output_parsers
    response = chain.invoke({"question": question})
    return response

st.title("ETEQnA - Ask Anything")


st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter your groq api key:",type="password")

llm=st.sidebar.selectbox("Select LLM Model",["Gemma-2-8b","Llama3-8b-8192","Gemma-9b-it"])
temperature=st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)
max_tokens=st.sidebar.slider("Max Tokens", 100, 300, 150)

st.write("Ask your question:")
user_input=st.text_input("You:")

if user_input:
    response=generate_answer(user_input,llm,api_key,temperature,max_tokens)
    st.write("Assistant:", response)
else:
    st.write("Please enter a question to get an answer.")






