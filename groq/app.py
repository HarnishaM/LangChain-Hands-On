import streamlit as st 
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time

load_dotenv()

#load the groq API key
groq_api_key=os.getenv('GROQ_API_KEY')

if "vector" not in st.session_state:
    st.session_state.embeddings=OllamaEmbeddings()
    st.session_state.loader=WebBaseLoader("https://peekaabookids.com/blogs/kidswear/10-short-panchatantra-stories-in-english-for-kids?srsltid=AfmBOopnmzrm4Itb03_3PtaFo1v2Z_9S7nZNNpYi5Niai79QihV8e8PB")
    st.session_state.web_docs=st.session_state.loader.load()
    
    st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    st.session_state.documents=st.session_state.text_splitter.split_documents(st.session_state.web_docs)
    
    st.session_state.db=FAISS.from_documents(st.session_state.documents,OllamaEmbeddings(model="nomic-embed-text"))

    
st.title("ChatGroq Demo")
llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="deepseek-r1-distill-llama-70b") 
    
prompt=ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    
    <context>
    {context}
    <context>
    Questions:{input}
    """    
)    

document_chain=create_stuff_documents_chain(llm,prompt)
retriever=st.session_state.db.as_retriever()
retriever_chain=create_retrieval_chain(retriever,document_chain)

prompt=st.text_input("Type your prompt here")

if prompt:
    start=time.process_time()
    response=retriever_chain.invoke({"input":prompt})
    print("Response time:" ,time.process_time()-start)
    st.write(response['answer'])
    
    #With a streamlit expander
    with st.expander("Document Similarity Search"):
        #Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("------------------------")
    
    