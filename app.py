import gradio as gr
import os
#import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# load environment variables from .env file (to get API key for Gemini)
load_dotenv()

# set model version
MODEL_NAME = 'gemini-1.5-flash'

# check to see if $PORT is set, and if so, set Gradio env var to use it. otherwise, use 8080 as default.
if "PORT" in os.environ:
  os.environ["GRADIO_SERVER_PORT"] = os.getenv(
    "PORT"
  )
else:
    os.environ["GRADIO_SERVER_PORT"] = "8080"
    
print(f"Setting Gradio server port to {os.getenv('GRADIO_SERVER_PORT')}")

# get model configured
# Configure the API key (replace with your actual key)
#genai.configure(api_key=os.getenv("API_KEY"))

#model = genai.GenerativeModel(model_name=MODEL_NAME)

# langchain model setup
os.environ["GOOGLE_API_KEY"] = os.getenv("API_KEY")

llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

def generate_response(message, history):
  messages = [
    (
        "system",
        "You are a helpful assistant that provides code examples",
    ),
    ("human", f"{message}"),
  ]
  ai_msg = llm.invoke(messages)
  return f"{ai_msg.content}"

interface = gr.ChatInterface(fn=generate_response, examples=["show me how to write a python script", "show me how to write a simple API in python", "Show me how to write a networking script in python"], title="Chat Bot")
interface.launch(server_name="0.0.0.0")