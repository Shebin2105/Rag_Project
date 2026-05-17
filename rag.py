import os
from dotenv import load_dotenv
import pandas as pd

#from faster_whisper import WhisperModel

#from langchain_core.documents import Document
from langchain_core.messages import (
    SystemMessage,
    HumanMessage
)

#from langchain_community.document_loaders import PyPDFLoader
#from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_groq import ChatGroq

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

# =========================
# LOAD ENV
# =========================

load_dotenv()
"""
# =========================
# WHISPER MODEL
# =========================

model = WhisperModel(
    "tiny",
    device="cpu",
    compute_type="int8"
)

print("Whisper model loaded")

# =========================
# TRANSCRIBE VIDEO
# =========================

segments, info = model.transcribe(
    "data/video.webm",
    beam_size=1
)

text = ""

for segment in segments:
    text += segment.text + " "

print("Transcription completed")

# =========================
# SAVE TRANSCRIPT
# =========================

with open("transcript.txt", "w", encoding="utf-8") as f:
    f.write(text)

# =========================
# CREATE DOCUMENT
# =========================

docs = [
    Document(
        page_content=text,
        metadata={
            "source": "youtube",
            "video_title": "Definition of AI",
            "video_id": "sNpME1MTiKg"
        }
    )
]

# =========================
# LOAD PDF
# =========================

loader = PyPDFLoader("data/ai.pdf")

pdf_pages = loader.load()

combined_docs = pdf_pages + docs

# =========================
# CHUNKING
# =========================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=200
)

chunked_docs = splitter.split_documents(combined_docs)

print(f"Total chunks: {len(chunked_docs)}")
"""
# =========================
# EMBEDDING MODEL
# =========================

embd = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large"
)

print("Embedding model loaded")

# =========================
# CREATE / LOAD CHROMA DB
# =========================

DB_PATH = "chroma_db"
"""
if not os.path.exists(DB_PATH):

    vectordb = Chroma.from_documents(
        documents=chunked_docs,
        embedding=embd,
        persist_directory=DB_PATH
    )

    vectordb.persist()

    print("New Vector DB created")
"""


vectordb = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embd
)

print("Existing Vector DB loaded")
# =========================
# LLM
# =========================

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=250
)

print("LLM loaded")

# =========================
# LANGGRAPH NODE
# =========================

def call_model(state: MessagesState):

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "Use three sentences maximum and keep the answer concise. "
        "Answer all questions to the best of your ability."
    )

    messages = [SystemMessage(content=system_prompt)] + state["messages"]

    response = llm.invoke(messages)

    return {"messages": response}

# =========================
# BUILD LANGGRAPH
# =========================

workflow = StateGraph(state_schema=MessagesState)

workflow.add_node("model", call_model)

workflow.add_edge(START, "model")

memory = MemorySaver()

app = workflow.compile(checkpointer=memory)

# =========================
# CHAT LOOP
# =========================

thread_id = "1"
