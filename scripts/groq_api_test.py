from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError(
           "GROQ_API_KEY is not set."
       )

llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
response = llm.invoke("Are you here?")
print(response.content) 