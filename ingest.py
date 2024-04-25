from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader, YoutubeLoader
import streamlit as st
import os
import ssl

ssl._create_default_https_context = ssl._create_stdlib_context


video_url = "https://www.youtube.com/watch?v=NM-zWTU7X-k"

DB_PATH = "/Users/ujwal_nischal/Desktop/LLM/Chat_with_Video/vectorstore"

model_name = "BAAI/bge-large-en-v1.5"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}

hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

def create_vector_db():
    embedding = hf
    # loader = PyPDFDirectoryLoader(DATA_PATH)
    loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=True)  # Corrected method name to 'from_url'
    doc = loader.load()
    print("Transcription completed for the YouTube video")
    # documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    # texts = text_splitter.split_documents(documents)
    texts = text_splitter.split_documents(doc)
    vectordb = FAISS.from_documents(documents=texts, embedding=embedding)
    vectordb.save_local(DB_PATH)
    print("Vector DB Successfully Created!")


create_vector_db()
