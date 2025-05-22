# kgrag.py
from typing import Annotated, List
from typing_extensions import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver
from neo4j import GraphDatabase
from dotenv import load_dotenv
import os
import re
import json

# Load environment variables
load_dotenv()

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Initialize LLM and tools
llm = ChatOpenAI(model="openai/o3-mini")
tavily_tool = TavilySearchResults(max_results=2)
tools = [tavily_tool]
llm_with_tools = llm.bind_tools(tools)

# Progress management
PROGRESS_FILE = "user_progress.json"

class State(TypedDict):
    messages: Annotated[list, add_messages]
    user_query: str

# Learning goals metadata
goal_metadata = {
    "LG1": {
        "description": "Utför strukturerade kognitiva bedömningar med MoCA och CID.",
        "key_concepts": ["MoCA", "CID", "kognitiv funktion", "vardagsliv"]
    },
    "LG2": {
        "description": "Tolka screeningresultat och koppla till påverkan på dagliga aktiviteter och delaktighet.",
        "key_concepts": ["screeningresultat", "daglig aktivitet", "kognitiva brister", "delaktighet"]
    },
    "LG3": {
        "description": "Genomför miljöbedömningar för att identifiera stödjande eller hindrande faktorer.",
        "key_concepts": ["miljöfaktorer", "funktion", "säkerhet", "delaktighet"]
    },
    "LG4": {
        "description": "Anpassa miljön för att stödja självständighet, säkerhet och orientering.",
        "key_concepts": ["anpassningar", "miljö", "självständighet", "orientering", "säkerhet"]
    },
    "LG5": {
        "description": "Rekommendera och introducera kognitiva hjälpmedel anpassade efter individens behov.",
        "key_concepts": ["hjälpmedel", "kognitivt stöd", "demensstadier", "behovsanalys"]
    },
    "LG6": {
        "description": "Planera och genomföra aktivitetsbaserade interventioner för självständighet och meningsfullhet.",
        "key_concepts": ["interventioner", "aktivitetsbaserat", "mening", "självständighet"]
    },
    "LG7": {
        "description": "Vägled anhöriga och personal i strategier för vardagsaktiviteter med personcentrerat fokus.",
        "key_concepts": ["anhörigstrategier", "personcentrerad vård", "anhöriga", "personal"]
    },
    "LG8": {
        "description": "Dokumentera arbetsterapiinsatser enligt ICF-ramverket.",
        "key_concepts": ["dokumentation", "ICF", "insats", "kommunikation"]
    },
    "LG9": {
        "description": "Skriv strukturerade remissvar baserade på funktionella bedömningar.",
        "key_concepts": ["remiss", "bedömning", "rapport", "demensutredning"]
    },
    "LG10": {
        "description": "Anpassa insatser utifrån sjukdomens progression från lindrig svikt till svår demens.",
        "key_concepts": ["insats", "progression", "lindrig svikt", "svår demens"]
    }
}

# Initialize Neo4j connection test
try:
    with driver.session() as session:
        msg = session.run("RETURN 'Neo4j ansluten' AS message").single()["message"]
        print("[✅ Neo4j]", msg)
except Exception as e:
    print("[❌ Neo4j FEL] Kunde inte ansluta:", e)

def load_progress() -> dict:
    """Load user progress from file"""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {f"LG{i}": "inte påbörjad" for i in range(1, 11)}

def save_progress(progress: dict):
    """Save user progress to file"""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)

def extract_entities_from_question(question: str) -> list[str]:
    """Extract entities from question using LLM with Neo4j schema knowledge"""
    all_node_types = []
    
    with driver.session() as session:
        result = session.run("CALL db.labels() YIELD label RETURN label")
        for record in result:
            all_node_types.append(record["label"])
    
    node_types_str = ", ".join(all_node_types)
    
    prompt = (
        f"Extract key entities from this question about dementia and its causes. "
        f"Our knowledge graph contains these node types: {node_types_str}. "
        f"For this question, identify terms that could represent nodes in our graph, "
        f"Return ONLY a comma-separated list with no additional text.\n\n"
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
    
    extracted_entities = [e.strip() for e in clean_response.split(",") if e.strip()]
    return extracted_entities

def query_neo4j_kg(user_input: str, max_entities=3) -> str:
    """Query Neo4j knowledge graph for relevant context"""
    entities = extract_entities_from_question(user_input)
    print(f"[KG-RAG] Extracted entities: {entities}")
    
    if not entities:
        return "No relevant entities found in the query."

    context_parts = []

    with driver.session() as session:
        for ent in entities:
            result = session.run(
                """
                MATCH (n)
                WHERE ANY(k IN keys(n) WHERE 
                    toLower(toString(n[k])) CONTAINS toLower($ent) OR
                    toLower($ent) CONTAINS toLower(toString(n[k])))
                WITH n
                MATCH path = (n)-[r*1..2]-(related)
                RETURN n, 
                       [rel in r | type(rel)] as relationship_types,
                       related
                LIMIT 15
                """,
                ent=ent
            )
            
            connections = []
            for record in result:
                source_node = record["n"]
                related_node = record["related"]
                rel_types = record["relationship_types"]
                
                source_props = dict(source_node)
                related_props = dict(related_node)
                
                def extract_node_name(props):
                    skip_props = ["<elementId>", "<id>", "id"]
                    for k, v in props.items():
                        if k not in skip_props and isinstance(v, str):
                            return v
                    for k, v in props.items():
                        if isinstance(v, str):
                            return v
                    return "Unknown"
                
                source_name = extract_node_name(source_props)
                related_name = extract_node_name(related_props)
                
                if len(rel_types) == 1:
                    connections.append(f"{source_name} {rel_types[0]} {related_name}")
                else:
                    connections.append(f"{source_name} is connected to {related_name} through {' and '.join(rel_types)}")
            
            if connections:
                context_parts.append(f"Entity '{ent}' context:")
                context_parts.extend(connections)
            else:
                context_parts.append(f"No connections found for '{ent}' in the knowledge graph.")

    return "\n".join(context_parts)

def kg_retriever(state: State):
    """Retrieve context from knowledge graph"""
    latest_user_message = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, dict) and msg.get("role") == "user":
            latest_user_message = msg["content"]
            break
        elif hasattr(msg, "type") and msg.type == "human":
            latest_user_message = msg.content
            break
    
    user_query = latest_user_message if latest_user_message else ""
    kg_context = query_neo4j_kg(user_query)
    
    return {
        "messages": state["messages"] + [
            SystemMessage(content=f"Knowledge Graph Context:\n{kg_context}")
        ],
        "user_query": user_query
    }

def web_retriever(state: State):
    """Retrieve context from web search"""
    user_query = state["user_query"]
    
    try:
        search_results = tavily_tool.invoke({"query": user_query})
        
        web_context = "Web Search Results:\n"
        for i, result in enumerate(search_results, 1):
            web_context += f"{i}. {result['title']}\n{result['content']}\n\n"
    except Exception as e:
        print(f"Tavily search error: {e}")
        web_context = "Web search failed or returned no results."
    
    return {
        "messages": state["messages"] + [
            SystemMessage(content=web_context)
        ],
        "user_query": user_query
    }

def chatbot(state: State):
    """Main chatbot function with KG and web context"""
    def get_latest_content(content_type):
        matching_messages = [
            msg.content for msg in state["messages"] 
            if isinstance(msg, SystemMessage) and content_type in msg.content
        ]
        return matching_messages[-1] if matching_messages else 'None'
    
    kg_context = get_latest_content("Knowledge Graph Context")
    web_results = get_latest_content("Web Search Results")

    print(f"Knowledge Graph Context: {kg_context} \n Web Search Results: {web_results}")
    
    system_prompt = f"""You are an AI tutor using Knowledge Graph-Enhanced RAG for dementia education.
    When answering, follow these principles:
    1. Connect concepts in a logical flow
    2. Ensure factual accuracy based on retrieved information
    3. Explain relationships between concepts clearly
    4. Remember details shared by the user in previous messages
    5. Respond in Swedish for Swedish queries, English for English queries
    
    Knowledge Graph Context: {kg_context}
    
    Web Search Results: {web_results}
    """
    
    message = llm_with_tools.invoke(
        state["messages"] + [SystemMessage(content=system_prompt)]
    )
    return {"messages": [message]}

# Build the graph
graph_builder = StateGraph(State)
graph_builder.add_node("kg_retriever", kg_retriever)
graph_builder.add_node("web_retriever", web_retriever)
graph_builder.add_node("chatbot", chatbot)

graph_builder.set_entry_point("kg_retriever")
graph_builder.add_edge("kg_retriever", "web_retriever")
graph_builder.add_edge("web_retriever", "chatbot")

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

def extract_keywords(user_input: str) -> List[str]:
    """Extract keywords from user input"""
    stopwords = {"vad", "är", "berätta", "om", "och", "kan", "du", "jag", "vill", "lära"}
    tokens = re.findall(r"\b\w+\b", user_input.lower())
    keywords = [w for w in tokens if w not in stopwords and len(w) > 2]
    return keywords if keywords else tokens

def match_input_to_goal(user_input: str) -> str | None:
    """Match user input to relevant learning goal"""
    kws = extract_keywords(user_input)
    scores = {
        gid: len(set(kws) & set(c.lower() for c in meta["key_concepts"])) 
        for gid, meta in goal_metadata.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None

def tutor_lesson(goal_id: str, level: str = "Nybörjare") -> str:
    """Generate lesson content for a specific goal"""
    desc = goal_metadata[goal_id]["description"]
    level_map = {"Nybörjare": "enkel", "Medel": "medel", "Expert": "avancerad"}
    complexity = level_map.get(level, "enkel")
    
    prompt = (
        f"Du är en handledare i demens och arbetsterapeutiska insatser. "
        f"Förklara detta steg-för-steg på {complexity} nivå:\n\n{desc}\n\n"
        f"Strukturera svaret med tydliga rubriker och konkreta exempel."
    )
    
    lesson = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    return lesson

def dynamic_quiz_list(goal_id: str, lesson_text: str) -> list[str]:
    """Generate quiz questions based on lesson content"""
    prompt = (
        f"Du är en arbetsterapeutisk handledare. Skapa 3 quizfrågor baserat på lektionen:\n"
        f"{lesson_text}\n\n"
        f"Märk frågorna som Q1, Q2, Q3. Returnera endast frågorna."
    )
    quiz_text = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    questions = re.findall(r"(Q[1-3][:\)]\s*.+?)(?=(?:\nQ[1-3]|$))", quiz_text, flags=re.S)
    return questions

def evaluate_answer(answer: str, question: str, lesson_text: str) -> str:
    """Evaluate student's answer to quiz question"""
    prompt = (
        f"Du utvärderar ett svar på svenska.\n\n"
        f"Lektion:\n{lesson_text}\n\n"
        f"Fråga:\n{question}\n\n"
        f"Svar:\n{answer}\n\n"
        "Om det visar förståelse, svara 'ja' följt av en uppmuntrande mening.\n"
        "Om inte, svara 'nej' och en kort mening om vad som saknas."
    )
    return llm.invoke([HumanMessage(content=prompt)]).content.strip()

def stream_graph_updates(user_input: str, conversation_history: list):
    """Process user input through the graph and return response"""
    try:
        human_message = HumanMessage(content=user_input)
        # Create a copy to avoid modifying the original
        processing_history = conversation_history.copy()
        processing_history.append(human_message)

        final_response = None
        for event in graph.stream(
            {"messages": processing_history},
            config={"thread_id": "default-thread"},
        ):
            for value in event.values():
                latest_message = value["messages"][-1]
                # Only return the final chatbot response, not system messages
                if hasattr(latest_message, 'content') and not isinstance(latest_message, SystemMessage):
                    final_response = latest_message.content
                
        return final_response if final_response else "Jag kunde inte generera ett svar just nu."
                
    except Exception as e:
        print(f"Error processing input: {e}")
        return f"Ett fel uppstod: {e}"