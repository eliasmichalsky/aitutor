from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
from neo4j import GraphDatabase
import os, re, json

# Load environment
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

try:
    with driver.session() as session:
        test = session.run("RETURN 'Neo4j connected' AS message").single()
        print("[✅ Neo4j] ", test["message"])
except Exception as e:
    print("[❌ Neo4j ERROR] Failed to connect:", e)

PROGRESS_FILE = "user_progress.json"
llm = ChatOpenAI(model="openai/o3-mini")

class State(TypedDict):
    messages: Annotated[list, add_messages]
    user_query: str

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {f"LG{i}": "not_started" for i in range(1, 11)}

def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)

user_progress = load_progress()
lesson_index = 1

goal_metadata = {
    "LG1": {
        "description": "Conduct structured cognitive assessments using MoCA and CID.",
        "key_concepts": ["MoCA", "CID", "cognitive function", "daily life"]
    },
    "LG2": {
        "description": "Interpret results from cognitive screening and connect them to impacts on daily activities and participation.",
        "key_concepts": ["screening results", "daily activity", "cognitive deficits", "participation"]
    },
    # Add LG3 to LG10 here if needed
}

def extract_keywords(user_input: str) -> list[str]:
    stopwords = {"what", "is", "tell", "me", "about", "the", "and", "can", "you", "i", "want", "to", "learn"}
    tokens = re.findall(r"\b\w+\b", user_input.lower())
    keywords = [word for word in tokens if word not in stopwords and len(word) > 2]
    print(f"[DEBUG] Tokens: {tokens}")
    print(f"[DEBUG] Keywords: {keywords}")
    return keywords if keywords else tokens

def query_neo4j_kg(user_input: str) -> str:
    keywords = extract_keywords(user_input)
    if not keywords:
        return "No searchable keywords found."
    triples, seen = [], set()
    with driver.session() as session:
        for keyword in keywords:
            result = session.run("""
            MATCH (n)
            WHERE any(prop IN keys(n) WHERE toLower(toString(n[prop])) CONTAINS toLower($query))
            WITH n
            MATCH path = (n)-[r*1..2]-(m)
            RETURN DISTINCT n, m, r
            LIMIT 50
            """, {"query": keyword})
            for record in result:
                node_a, node_b, rels = record["n"], record["m"], record["r"]
                a_name = node_a.get("name") or node_a.get("title") or "[Unnamed Node]"
                b_name = node_b.get("name") or node_b.get("title") or "[Unnamed Node]"
                a_type = list(node_a.labels)[0] if node_a.labels else "Unknown"
                b_type = list(node_b.labels)[0] if node_b.labels else "Unknown"
                rel_types = " → ".join(set([rel.type for rel in rels]))
                line = f"{a_name} ({a_type}) —[{rel_types}]→ {b_name} ({b_type})"
                if line not in seen:
                    triples.append(line)
                    seen.add(line)
    return f"Knowledge Graph Context for '{user_input}':\n" + "\n".join(triples) if triples else f"No knowledge graph context found for '{user_input}'."

def dynamic_quiz(goal_id):
    if goal_id not in goal_metadata:
        print(f"⚠️ No quiz available for {goal_id}")
        return
    meta = goal_metadata[goal_id]
    description, concepts = meta["description"], ", ".join(meta["key_concepts"])
    prompt = f"You are an occupational therapy tutor. Create a question that tests understanding of this goal:\n'{description}'\nExpected concepts: {concepts}\nReturn only the question."
    question = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    print(f"\n📘 {question}")
    user_answer = input("Your answer: ")
    eval_prompt = f"Evaluate: '{user_answer}'\nShould show knowledge of: {concepts}.\nDoes the answer show full understanding? Reply 'yes' or 'no' with 1 line of feedback."
    result = llm.invoke([HumanMessage(content=eval_prompt)]).content.strip()
    print("\n🧠 Feedback:", result)
    if "yes" in result.lower():
        user_progress[goal_id] = "mastered"
        print(f"🎉 You've mastered {goal_id}!")
    else:
        user_progress[goal_id] = "in_progress"
        print(f"🔄 Keep working on {goal_id}.")
    save_progress(user_progress)

def tutor_lesson(goal_id):
    meta = goal_metadata[goal_id]
    description = meta["description"]
    prompt = f"You are a dementia tutor. Teach this concept step by step in simple language, like explaining it to a student:\n{description}"
    response = llm.invoke([HumanMessage(content=prompt)]).content
    print("\n📖 Lesson:")
    print(response)
    print("\n💬 You can ask follow-up questions, or type 'quiz' to test your understanding.")

def start_lesson_sequence():
    global lesson_index
    while lesson_index <= 10:
        goal_id = f"LG{lesson_index}"
        if user_progress[goal_id] == "mastered":
            lesson_index += 1
            continue
        tutor_lesson(goal_id)
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() == "quiz":
                dynamic_quiz(goal_id)
                if user_progress[goal_id] == "mastered":
                    lesson_index += 1
                    break
            else:
                answer = llm.invoke([HumanMessage(content=user_input)]).content
                print("Tutor:", answer)
    print("\n✅ All kursmål completed!")

# Launch tutor
if __name__ == "__main__":
    print("Welcome to the AI Dementia Tutor! Type 'start' to begin your learning journey.")
    while True:
        user_input = input(">>> ").strip().lower()
        if user_input == "start":
            start_lesson_sequence()
            break
        elif user_input == "progress":
            for k, v in user_progress.items():
                print(f"{k}: {v}")
        elif user_input == "exit":
            break
        else:
            print("Type 'start' to begin or 'exit' to quit.")
