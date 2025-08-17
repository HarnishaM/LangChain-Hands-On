import requests 
import streamlit as st 

def get_gemini_response(input_text):
    response=requests.post("http://localhost:8000/essay/invoke",
    json={'input':{'topic':input_text}})
    
    return response.json()['output']['content']

def get_ollama_response(input_text):
    response=requests.post("http://localhost:8000/poem/invoke",
    json={'input':{'topic':input_text}})
    
    # data = response.json()
    # print("DEBUG RESPONSE:", data)  # 👈 check the actual structure
    # return data
    
    return response.json()['output']


#streamlit framework
st.title('Langchaim Demo with Qwen API')
input_text=st.text_input("Write an essay on")
input_text1=st.text_input("Write a poem on")

if input_text:
    st.write(get_gemini_response(input_text))
    
if input_text1:
    st.write(get_ollama_response(input_text1))