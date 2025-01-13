import streamlit as st
from langchain.schema import HumanMessage, AIMessage
from react_agent import create_react_agent
from st_callback_util import invoke_our_graph
import asyncio


if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
if "agent" not in st.session_state:
    agent = create_react_agent()
    st.session_state["agent"] = agent


for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])


def display_msg(msg, author):
    st.session_state.messages.append({"role": author, "content": msg})
    st.chat_message(author).write(msg)

# if user_query:
if prompt := st.chat_input():
    display_msg(prompt, 'user')
    with st.chat_message("assistant"):
        placeholder = st.container()
        response = asyncio.run(invoke_our_graph(st.session_state.messages, placeholder))
        st.session_state.messages.append({"role": "assistant", "content": response})

if st.sidebar.button("Reset chat history"):
    st.session_state.messages = []