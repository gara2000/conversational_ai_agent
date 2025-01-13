from langgraph.graph import StateGraph, START
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

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
system_message = SystemMessage(content="You are a helpful assistant that help the user to plan an optimized itinerary.")

# def create_react_agent(llm_with_tools, tools, memory=None, system_message="", verbose=True):
def create_react_agent():
  # Assistant node
  def assistant(state: MessagesState, config: RunnableConfig) -> MessagesState:
      response = llm_with_tools.invoke([system_message] + state["messages"], config)
      return {"messages": [response]}
  builder = StateGraph(MessagesState)
  builder.add_node("assistant", assistant)
  builder.add_node("tools", ToolNode(tools))
  builder.add_edge(START, "assistant")
  builder.add_conditional_edges(
      "assistant",
      # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
      # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
      tools_condition,
  )
  builder.add_edge("tools", "assistant")
  if memory:
    graph = builder.compile(checkpointer=memory)
  else:
    graph = builder.compile()
#   if verbose:
#     display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
  return graph