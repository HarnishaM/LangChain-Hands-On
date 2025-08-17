#Creating all the APIs
from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
#from langchain.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langserve import add_routes
import uvicorn
import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv

load_dotenv()

os.environ["GOOGLE_API_KEY"]=os.getenv("GEMINI_API_KEY")

app=FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API Server"
)

add_routes(
    app,
    ChatGoogleGenerativeAI(model="gemini-1.5-flash"),
    path="/gemini"
)

model=ChatGoogleGenerativeAI(model="gemini-1.5-flash")
#ollama qwen
llm=Ollama(model="qwen:0.5b")

prompt1=ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words")
prompt2=ChatPromptTemplate.from_template("Write me a poem about {topic} with 100 words")

add_routes(
    app,
    prompt1|model,
    path="/essay"
)

add_routes(
    app,
    prompt2|llm,
    path="/poem"
)

if __name__=="__main__":
    uvicorn.run(app,host="localhost" ,port=8000)
