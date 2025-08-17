from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st 
import os 
from dotenv import load_dotenv

#langsmith tracking
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

prompt=ChatPromptTemplate.from_messages(
    [
        ("system", "You are a polite and helpful assistant. Please respond to the user queries"),
        ("user","Question:{question}")
    ]
)

#streamlit framework-- basically ui of the chatbot
st.title('Langchain Demo with QWEN API')
input_text=st.text_input("Search the topic you want")

#ollama LLM -- brain
llm = Ollama(model="qwen:0.5b")
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({'question':input_text}))
