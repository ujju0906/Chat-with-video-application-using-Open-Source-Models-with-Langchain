import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain import PromptTemplate
from langchain_community.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# App config
st.set_page_config(page_title="LAM Knowledge Bot", page_icon="🤖")
st.title("chat with video")

import os
os.environ['CURL_CA_BUNDLE'] = ''

DATA_PATH = ""
DB_PATH = "/Users/ujwal_nischal/Desktop/LLM/Chat_with_Video/vectorstore"

model_name = "BAAI/bge-large-en-v1.5"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}

hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

def get_response(user_query):
    prompt_template = """ 
        Use the following pieces of information to answer the user's question.The given context is a video transcript perform good analysis of these transcripts. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer. 
        Context: {context} 
        Question: {question} 
        Only return the helpful answer. Answer must be detailed and well explained. 
        Helpful answer: 
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm = Ollama(
        base_url="http://localhost:11434",
        model="mistral",
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )

    # Load the FAISS index
    new_db = FAISS.load_local(DB_PATH, hf, allow_dangerous_deserialization=True)

    # Initialize the retriever
    retriever = new_db.as_retriever(search_kwargs={"k": 4})

    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

    # Initialize the QA model
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
        verbose=True
    )

    response = qa(user_query)
    answer = response['result']
    return answer

# Session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hello, I am a bot. How can I help you?")]

# Conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# User input handling
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("Human"):
        st.markdown(user_query)
    with st.chat_message("AI"):
        response = get_response(user_query)
        st.write(response)
    st.session_state.chat_history.append(AIMessage(content=response))
