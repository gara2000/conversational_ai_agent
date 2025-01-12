from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

def create_react_agent(llm_with_tools, tools, memory=None, system_message="", verbose=True):
  # Assistant node
  def assistant(state: MessagesState) -> MessagesState:
      response = llm_with_tools.invoke([system_message] + state["messages"])
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