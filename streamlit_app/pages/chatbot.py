import streamlit as st
import random
import time
import pymongo
from pymongo import MongoClient
from dotenv import load_dotenv
import pandas as pd
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from llm.llm import LLM

st.set_page_config(page_title="Chatbot", page_icon="../images/logo.png")
a = LLM()

# Streamed response emulator
def response_generator(question):
    response = a.generate_answer(question)
    # Emulate delay
    for word in response.split():
        yield word + " "
        time.sleep(0.10)

def save_data_to_csv(df):
    df.to_csv("data/documents.csv", encoding='utf-8', index=False)

def clear_chat():
    st.session_state.messages = []
    st.rerun()

st.markdown(
    """
        <style>
            .st-emotion-cache-janbn0 {
                flex-direction: row-reverse;
                text-align: justify;
                text-align-last: right;
                -moz-text-align-last: right;
            }
            .st-emotion-cache-1c7y2kd {
                flex-direction: row-reverse;
                text-align: justify;
                text-align-last: right;
                -moz-text-align-last: right;
            }
            .st-emotion-cache-4oy321 {
                padding: 1rem;
            }
             .st-emotion-cache-13ln4jf {
                height: 100vh;
            }
        </style>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

messages = st.session_state.messages # Current chat messages

tab1, tab2 = st.tabs(["About", "Chat"])
with tab1:
    df = pd.read_csv('data/documents.csv')

    st.header("Our Large Language Model")
    st.write("Our Large Language Model is created to help you with your questions. It is trained on a large dataset of documents and can answer a wide range of questions.")
    st.subheader("Document Loading")

    st.write("Below you can see the most common words from the documents saved and used for training the model.")
    st.image('data/wordcloud.png', caption='common words')
    st.dataframe(df.head(10))
    
with tab2:
    # clear buttons
    col1, col2 = st.columns([9, 1])

    with col2:
        if st.button("Clear", type="primary"):
            clear_chat()
    # Display chat messages
    container = st.container(height=645, border=False)
    with container:
    # First message from chatbot
        first_message = "Hello! How can I help you today?"
        if "assistant" not in [message["role"] for message in messages]:
            messages.append({"role": "assistant", "content": first_message})
            
        # Display chat messages
        for message in messages:
            with container.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat message input
    if prompt := st.chat_input("Type your message",key="new_chat"):
    # Save user message to session state
        messages.append({"role": "user", "content": prompt})
        with container.chat_message("user"):
            st.markdown(prompt)
        # Generate chatbot response and save to session state
        with container.chat_message("assistant"):
            response = st.write_stream(response_generator(prompt))
        messages.append({"role": "assistant", "content": response})