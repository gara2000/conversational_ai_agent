import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langgraph.checkpoint.memory import MemorySaver
from react_agent import create_react_agent

llm = ChatOpenAI(model="gpt-4o-mini")
text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200
        )
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
raw_documents = PyPDFLoader('italy_travel.pdf').load()
documents = text_splitter.split_documents(raw_documents)
db = FAISS.from_documents(documents, embeddings)
websearch_tool = TavilySearchResults(max_results=2)
retriever_tool = create_retriever_tool(
    db.as_retriever(), 
    "italy_travel",
    "Searches and returns documents regarding touristic places in Italy such as the Pantheon."
)
tools = [retriever_tool, websearch_tool]
llm_with_tools = llm.bind_tools(tools)
memory = MemorySaver()
sys_message = SystemMessage(content="You are a helpful assistant that help the user to plan an optimized itinerary.")
agent = create_react_agent(llm_with_tools, tools, system_message=sys_message)

user_query = st.text_input(
    "**Where are you planning your next vacation?**",
    placeholder="Ask me anything!"
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
if "memory" not in st.session_state:
    st.session_state['memory'] = memory


for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])


def display_msg(msg, author):
    st.session_state.messages.append({"role": author, "content": msg})
    st.chat_message(author).write(msg)

if user_query:
    display_msg(user_query, 'user')
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())
        new_state = agent.invoke({"messages": HumanMessage(content=user_query)}, config={"callbacks": [st_cb], "configurable": {"thread_id": "0"}})
        response = new_state["messages"][-1].content
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)

if st.sidebar.button("Reset chat history"):
    st.session_state.messages = []