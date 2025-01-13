import streamlit as st
from langchain.schema import HumanMessage
from react_agent import create_react_agent, invoke_our_graph
from st_callback_util import get_streamlit_cb


if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
if "agent" not in st.session_state:
    # agent = create_react_agent(llm_with_tools, tools, system_message=sys_message, memory=memory)
    agent = create_react_agent()
    st.session_state["agent"] = agent


for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])


def display_msg(msg, author):
    st.session_state.messages.append({"role": author, "content": msg})
    st.chat_message(author).write(msg)

if prompt := st.chat_input():
    display_msg(prompt, 'user')
    with st.chat_message("assistant"):
        st_callback = get_streamlit_cb(st.container())
        response = invoke_our_graph(st.session_state["agent"], st.session_state.messages, [st_callback])
        # response = st.session_state["agent"].invoke({"messages": HumanMessage(content=prompt)}, config={"callbacks": [st_cb], "configurable": {"thread_id": "0"}})
        ai_msg = response["messages"][-1].content
        st.session_state.messages.append({"role": "assistant", "content": ai_msg})

if st.sidebar.button("Reset chat history"):
    st.session_state.messages = []