import gradio as gr
import os

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

def echo(message, history):
    return message

interface = gr.ChatInterface(fn=echo, examples=["show me how to write a python script", "show me how to write a simple API in python", "Show me how to write a networking script in python"], title="Echo Bot")
interface.launch(server_name="0.0.0.0")