import streamlit as st
import random
import time
import pymongo
from pymongo import MongoClient
from dotenv import load_dotenv
import pandas as pd
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../../'))
import llm.llm as llm

st.set_page_config(page_title="Chatbot", page_icon="../img/logo.png")
llm = llm.LLM()

# Streamed response emulator
def response_generator(question):
    response = llm.generate_answer(question)
    # Emulate delay
    for word in response.split():
        yield word + " "
        time.sleep(0.10)

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
# Save and clear buttons
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