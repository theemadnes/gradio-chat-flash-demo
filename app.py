import gradio as gr
import os
#import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

from langchain_google_vertexai import VertexAIEmbeddings, VertexAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_google_alloydb_pg import (
    AlloyDBEngine,
    AlloyDBVectorStore,
    AlloyDBChatMessageHistory,
)

# load environment variables from .env file (to get API key for Gemini)
load_dotenv()

PROJECT_ID = os.environ.get('PROJECT_ID')  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}
cluster_name = os.environ.get('DB_CLUSTER')  # @param {type:"string"}
instance_name = os.environ.get('DB_INSTANCE')  # @param {type:"string"}
database_name = "ainetdevopsdb"  # @param {type:"string"}
vector_table_name = "code_vector_table"
password = os.environ.get('DB_PASS')

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

# Intialize the embedding service
embeddings_service = VertexAIEmbeddings(
    model_name="textembedding-gecko@003", project=PROJECT_ID
)


engine = AlloyDBEngine.from_instance(
    project_id=PROJECT_ID,
    instance=instance_name,
    region=LOCATION,
    cluster=cluster_name,
    database=database_name,
    user="postgres",
    password=password,
)

# Intialize the Vector Store
vector_store = AlloyDBVectorStore.create_sync(
    engine=engine,
    embedding_service=embeddings_service,
    table_name=vector_table_name,
    metadata_columns=[
        "filenames",
        "languages",
    ],
)

message_table_name = "message_store"

engine.init_chat_history_table(table_name=message_table_name)

chat_history = AlloyDBChatMessageHistory.create_sync(
    engine,
    session_id="test-session",
    table_name=message_table_name,
)

# Prepare some prompt templates for the ConversationalRetrievalChain
prompt = PromptTemplate(
    template="""Use all the information from the context and the conversation history to answer new question. If you see the answer in previous conversation history or the context. \
Answer it with clarifying the source information. If you don't see it in the context or the chat history, just say you \
didn't find the answer in the given data. Don't make things up. Add any citations of filenames that were referenced as the answer.

Previous conversation history from the questioner. "Human" was the user who's asking the new question. "Assistant" was you as the assistant:
```{chat_history}
```

Vector search result of the new question:
```{context}
```

New Question:
```{question}```

Answer:""",
    input_variables=["context", "question", "chat_history"],
)
condense_question_prompt_passthrough = PromptTemplate(
    template="""Repeat the following question:
{question}
""",
    input_variables=["question"],
)


retriever = vector_store.as_retriever(
    search_type="mmr", search_kwargs={"k": 5, "lambda_mult": 0.8}
)    

# langchain model setup
os.environ["GOOGLE_API_KEY"] = os.environ.get("API_KEY")

llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

chat_history.clear()

memory = ConversationSummaryBufferMemory(
    llm=llm,
    chat_memory=chat_history,
    output_key="answer",
    memory_key="chat_history",
    return_messages=True,
)

rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    verbose=False,
    memory=memory,
    condense_question_prompt=condense_question_prompt_passthrough,
    combine_docs_chain_kwargs={"prompt": prompt},
)

def generate_response(message, history):
  ans = rag_chain({"question": message, "chat_history": chat_history})["answer"]
  return ans
  '''messages = [
    (
        "system",
        "You are a helpful assistant that provides code examples",
    ),
    ("human", f"{message}"),
  ]
  ai_msg = llm.invoke(messages)
  return f"{ai_msg.content}"'''

interface = gr.ChatInterface(fn=generate_response, examples=["show me how to write a python script", "show me how to write a simple API in python", "Show me how to write a networking script in python"], title="Chat Bot")
interface.launch(server_name="0.0.0.0")