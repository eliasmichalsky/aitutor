from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import SystemMessage
from langchain_core.messages import HumanMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI

from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool

from neo4j import GraphDatabase

from dotenv import load_dotenv
import os

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

class State(TypedDict):
    # Messages have the type "list". The add_messages function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]
    user_query: str

def query_neo4j_kg(user_input: str, max_entities=3) -> str:
    entities = extract_entities_from_question(user_input)
    print(f"[KG-RAG] Extracted entities: {entities}")
    
    if not entities:
        return "No relevant entities found in the query."

    context_parts = []

    with driver.session() as session:
        for ent in entities:
            # First find nodes that match our entity
            result = session.run(
                """
                MATCH (n)
                WHERE toLower(n.name) CONTAINS toLower($ent) OR toLower(n.title) CONTAINS toLower($ent)
                WITH n
                // For each matching node, find connected nodes up to 2 hops away
                MATCH path = (n)-[r*1..2]-(related)
                // Return the node, relationship types, and related nodes
                RETURN n, 
                       [rel in r | type(rel)] as relationship_types,
                       related
                LIMIT 15
                """,
                ent=ent
            )
            
            # Process the results
            connections = []
            for record in result:
                source_node = record["n"]
                related_node = record["related"]
                rel_types = record["relationship_types"]
                
                # Get node names/titles
                source_name = source_node.get("name") or source_node.get("title") or "Unknown"
                related_name = related_node.get("name") or related_node.get("title") or "Unknown"
                
                # For 1-hop relationships
                if len(rel_types) == 1:
                    connections.append(f"{source_name} {rel_types[0]} {related_name}")
                # For 2-hop relationships
                else:
                    connections.append(f"{source_name} is connected to {related_name} through {' and '.join(rel_types)}")
            
            if connections:
                context_parts.append(f"Entity '{ent}' context:")
                context_parts.extend(connections)
            else:
                context_parts.append(f"No connections found for '{ent}' in the knowledge graph.")

    return "\n".join(context_parts)



def extract_entities_from_question(question: str) -> list[str]:
    prompt = (
        "Extract the key movie-related entities from the following question. "
        "Return ONLY a comma-separated list of movie titles, actor names, or other related terms "
        "with no additional text, prefixes, or formatting.\n\n"
        f"Question: {question}"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    
    clean_response = response.content

    if "•" in clean_response or "\n-" in clean_response:
        items = []
        for line in clean_response.split("\n"):
            if "•" in line:
                items.append(line.split("•")[1].strip())
            elif line.strip().startswith("-"):
                items.append(line.strip()[1:].strip())
        clean_response = ", ".join(items)
    return [e.strip() for e in clean_response.split(",") if e.strip()]


tool = TavilySearchResults(max_results=2)
tools = [tool]
llm = ChatOpenAI(model="openai/o3-mini")
llm_with_tools = llm.bind_tools(tools)

def kg_retriever(state: State):
    user_query = state["messages"][-1].content
    kg_context = query_neo4j_kg(user_query)
    return {
        "messages": [
            SystemMessage(content=f"Knowledge Graph Context:\n{kg_context}")
        ],
        "user_query": user_query  # Pass the original query forward
    }

def web_retriever(state: State):
    # Use the original user query, not the KG context
    user_query = state["user_query"]
    
    try:
        # Make sure the Tavily API is properly configured
        search_results = tool.invoke({"query": user_query})
        
        # Format the search results
        web_context = "Web Search Results:\n"
        for i, result in enumerate(search_results, 1):
            web_context += f"{i}. {result['title']}\n{result['content']}\n\n"
    except Exception as e:
        # Handle errors gracefully
        print(f"Tavily search error: {e}")
        web_context = "Web search failed or returned no results."
    
    # Keep the KG context from the previous step
    return {
        "messages": state["messages"] + [
            SystemMessage(content=web_context)
        ],
        "user_query": user_query
    }

def chatbot(state: State):
    # The LLM now has access to both KG context and web search results
    kg_context = [msg for msg in state["messages"] if "Knowledge Graph Context" in msg.content]
    web_context = [msg for msg in state["messages"] if "Web Search Results" in msg.content]
    user_query = state["user_query"]
    
    system_prompt = f"""You are an AI tutor using Knowledge Graph-Enhanced RAG.
    When answering, follow these principles:
    1. Connect concepts in a logical flow
    2. Ensure factual accuracy based on retrieved information
    3. Explain relationships between concepts clearly
    
    Knowledge Graph Context: {kg_context[-1].content if kg_context else 'None'}

    Web Search Results: {web_context[-1].content if web_context else 'None'}
    """
    
    messages = [SystemMessage(content=system_prompt)] + [HumanMessage(content=user_query)]
    message = llm_with_tools.invoke(messages)
    return {"messages": [message]}

# Update the graph structure
graph_builder = StateGraph(State)

# Add nodes
graph_builder.add_node("kg_retriever", kg_retriever)
graph_builder.add_node("web_retriever", web_retriever)
graph_builder.add_node("chatbot", chatbot)

# Set entry and flow
graph_builder.set_entry_point("kg_retriever")
graph_builder.add_edge("kg_retriever", "web_retriever")
graph_builder.add_edge("web_retriever", "chatbot")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)


def stream_graph_updates(user_input: str):
    for event in graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config={"thread_id": "default-thread"},
    ):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


while True:
    try:
        user_input = input("User: ")
        stream_graph_updates(user_input)
    except:
        break