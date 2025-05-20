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

goal_metadata = {
    "LG1": {
        "description": "Conduct structured cognitive assessments using MoCA and CID.",
        "key_concepts": ["MoCA", "CID", "cognitive function", "daily life"]
    },
    "LG2": {
        "description": "Interpret results from cognitive screening and connect them to impacts on daily activities and participation.",
        "key_concepts": ["screening results", "daily activity", "cognitive deficits", "participation"]
    },
    "LG3": {
        "description": "Perform systematic environmental assessments to identify supportive or hindering factors.",
        "key_concepts": ["environmental factors", "function", "safety", "participation"]
    },
    "LG4": {
        "description": "Implement individualized environmental adaptations that support independence, safety, and orientation.",
        "key_concepts": ["adaptations", "environment", "independence", "orientation", "safety"]
    },
    "LG5": {
        "description": "Prescribe and introduce cognitive support tools and assistive technology tailored to the individual's needs.",
        "key_concepts": ["assistive technology", "cognitive support", "dementia stages", "needs assessment"]
    },
    "LG6": {
        "description": "Plan and deliver activity-based interventions that promote independence and meaningfulness.",
        "key_concepts": ["interventions", "activity-based", "meaning", "independence"]
    },
    "LG7": {
        "description": "Coach relatives and staff in strategies to support everyday activities for people with dementia, with a person-centered care focus.",
        "key_concepts": ["caregiver strategies", "person-centered care", "relatives", "staff"]
    },
    "LG8": {
        "description": "Document occupational therapy interventions in the dementia field using the ICF framework.",
        "key_concepts": ["documentation", "ICF", "intervention", "communication"]
    },
    "LG9": {
        "description": "Write structured referral responses based on functional and activity assessments to support basic dementia investigations.",
        "key_concepts": ["referral", "assessment", "report", "dementia investigation"]
    },
    "LG10": {
        "description": "Adapt occupational therapy interventions based on the progression of the disease from mild cognitive impairment to severe dementia.",
        "key_concepts": ["intervention", "progression", "mild cognitive impairment", "severe dementia"]
    }
}

def extract_keywords(user_input: str) -> list[str]:
    stopwords = {"what", "is", "tell", "me", "about", "the", "and", "can", "you", "i", "want", "to", "learn"}
    tokens = re.findall(r"\b\w+\b", user_input.lower())
    keywords = [word for word in tokens if word not in stopwords and len(word) > 2]
    return keywords if keywords else tokens

def match_input_to_goal(user_input: str) -> str | None:
    keywords = extract_keywords(user_input)
    matches = {}

    for goal_id, meta in goal_metadata.items():
        concepts = [c.lower() for c in meta["key_concepts"]]
        overlap = set(keywords) & set(concepts)
        if overlap:
            matches[goal_id] = len(overlap)

    if not matches:
        return None

    best_match = max(matches, key=matches.get)
    return best_match

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


def dynamic_quiz(goal_id, lesson_content):
    if goal_id not in goal_metadata:
        print(f"⚠️ No quiz available for {goal_id}")
        return

    # Generate questions based only on what was taught in the lesson
    prompt = (
        f"You are an occupational therapy tutor. Based ONLY on the following lesson content, create a 3-question quiz to test understanding:\n\n"
        f"{lesson_content}\n\n"
        f"The quiz should:\n"
        f"- Ask only about information clearly explained in the lesson.\n"
        f"- Include a mix of basic recall, application, and reasoning.\n"
        f"- Label the questions Q1, Q2, Q3.\n"
        f"- Return only the questions."
    )

    questions_text = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    questions = re.findall(r"(Q\d[:\.\)]\s*)(.+?)(?=\nQ\d[:\.\)]|\Z)", questions_text, re.DOTALL)

    if not questions or len(questions) < 3:
        print("❌ Failed to generate structured quiz questions.")
        return

    meta = goal_metadata[goal_id]
    concepts = ", ".join(meta["key_concepts"])
    correct_answers = 0
    total = len(questions)

    for label, question in questions:
        print(f"\n📘 {label.strip()} {question.strip()}")
        user_answer = input("Your answer: ").strip()

        eval_prompt = (
            f"You are evaluating a student's answer to a quiz question based on a lesson.\n\n"
            f"Lesson content:\n{lesson_content}\n\n"
            f"Question:\n{question.strip()}\n\n"
            f"Student's answer:\n{user_answer.strip()}\n\n"
            f"Please do the following:\n"
            f"1. Decide if the student's answer shows understanding of the core idea (even if the wording differs).\n"
            f"2. If it does, reply with 'yes' and one sentence of encouraging feedback.\n"
            f"3. If it does not, reply with 'no' and one short sentence explaining what was missing or unclear.\n"
            f"Be generous when crediting correct answers. Focus on meaning over exact phrasing."
        )


        result = llm.invoke([HumanMessage(content=eval_prompt)]).content.strip()
        print("🧠 Feedback:", result)

        if result.lower().startswith("yes"):
            correct_answers += 1

    print(f"\n✅ You answered {correct_answers} out of {total} correctly.")

    if correct_answers >= 2:
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

def run_goal_session(goal_id):
    meta = goal_metadata[goal_id]
    description = meta["description"]

    # Generate lesson and store it
    conversation = [
        HumanMessage(content=f"You are a dementia tutor. Teach this concept step by step in simple language:\n{description}")
    ]
    lesson_content = llm.invoke(conversation).content.strip()

    print("\n📖 Lesson:")
    print(lesson_content)
    print("\n💬 You can ask follow-up questions, or type 'quiz' to test your understanding.")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "quiz":
            dynamic_quiz(goal_id, lesson_content)
            print("\n👈 Returning to the learning goals menu...")
            return
        elif user_input.lower() in {"exit", "menu", "mål", "next"}:
            print("\n👈 Returning to the learning goals menu...")
            return
        else:
            conversation.append(HumanMessage(content=user_input))
            reply = llm.invoke(conversation).content
            print("Tutor:", reply)



def list_goals():
    print("\n📋 Available Learning Goals:")
    available = False
    for key, data in goal_metadata.items():
        status = user_progress.get(key, "not_started")
        if status != "mastered":
            symbol = {"not_started": "⬜", "in_progress": "🟡"}.get(status, "⬜")
            print(f"{symbol} {key}: {data['description']}")
            available = True
    if not available:
        print("🎉 All goals have been mastered!")

def goal_selection_loop():
    while True:
        list_goals()
        user_input = input("\n>>> ").strip().upper()
        if user_input == "EXIT":
            break
        elif user_input == "PROGRESS":
            for k, v in user_progress.items():
                print(f"{k}: {v}")
        elif user_input in goal_metadata and user_progress[user_input] != "mastered":
            run_goal_session(user_input)
        else:
            matched_goal = match_input_to_goal(user_input)
            if matched_goal and user_progress[matched_goal] != "mastered":
                print(f"🔍 Matched to goal: {matched_goal} - {goal_metadata[matched_goal]['description']}")
                run_goal_session(matched_goal)
            else:
                print("❌ Could not match your input to a learning goal or it is already mastered. Try again.")

# Launch tutor
if __name__ == "__main__":
    print("🎓 Welcome to the AI Dementia Tutor!")
    goal_selection_loop()
