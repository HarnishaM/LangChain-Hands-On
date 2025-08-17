from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st 
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GOOGLE_API_KEY"]=os.getenv("GEMINI_API_KEY")
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
st.title('Langchain Demo with GEMINI API')
input_text=st.text_input("Search the topic you want")

#GEMINI LLM -- brain
from langchain.chat_models import init_chat_model

llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({'question':input_text}))
    
    