from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from neo4j import GraphDatabase
import os, re, json

# Load environment variables
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Initialize Neo4j driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
try:
    with driver.session() as session:
        msg = session.run("RETURN 'Neo4j connected' AS message").single()["message"]
        print("[✅ Neo4j]", msg)
except Exception as e:
    print("[❌ Neo4j ERROR] Failed to connect:", e)

# Progress file and LLM init
PROGRESS_FILE = "user_progress.json"
llm = ChatOpenAI(model="openai/o3-mini")

class State(TypedDict):
    messages: Annotated[list, add_messages]
    user_query: str

# Load / Save progress
def load_progress() -> dict:
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {f"LG{i}": "not_started" for i in range(1, 11)}

def save_progress(progress: dict):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)

user_progress = load_progress()

# Learning goals metadata
goal_metadata = {
    "LG1": {"description": "Conduct structured cognitive assessments using MoCA and CID.", "key_concepts": ["MoCA", "CID", "cognitive function", "daily life"]},
    "LG2": {"description": "Interpret results from cognitive screening and connect them to impacts on daily activities and participation.", "key_concepts": ["screening results", "daily activity", "cognitive deficits", "participation"]},
    "LG3": {"description": "Perform systematic environmental assessments to identify supportive or hindering factors.", "key_concepts": ["environmental factors", "function", "safety", "participation"]},
    "LG4": {"description": "Implement individualized environmental adaptations that support independence, safety, and orientation.", "key_concepts": ["adaptations", "environment", "independence", "orientation", "safety"]},
    "LG5": {"description": "Prescribe and introduce cognitive support tools and assistive technology tailored to the individual's needs.", "key_concepts": ["assistive technology", "cognitive support", "dementia stages", "needs assessment"]},
    "LG6": {"description": "Plan and deliver activity-based interventions that promote independence and meaningfulness.", "key_concepts": ["interventions", "activity-based", "meaning", "independence"]},
    "LG7": {"description": "Coach relatives and staff in strategies to support everyday activities for people with dementia, with a person-centered care focus.", "key_concepts": ["caregiver strategies", "person-centered care", "relatives", "staff"]},
    "LG8": {"description": "Document occupational therapy interventions in the dementia field using the ICF framework.", "key_concepts": ["documentation", "ICF", "intervention", "communication"]},
    "LG9": {"description": "Write structured referral responses based on functional and activity assessments to support basic dementia investigations.", "key_concepts": ["referral", "assessment", "report", "dementia investigation"]},
    "LG10": {"description": "Adapt occupational therapy interventions based on the progression of the disease from mild cognitive impairment to severe dementia.", "key_concepts": ["intervention", "progression", "mild cognitive impairment", "severe dementia"]}
}

# Text processing utilities
def extract_keywords(user_input: str) -> list[str]:
    stopwords = {"what","is","tell","me","about","the","and","can","you","i","want","to","learn"}
    tokens = re.findall(r"\b\w+\b", user_input.lower())
    keywords = [w for w in tokens if w not in stopwords and len(w)>2]
    return keywords if keywords else tokens


def match_input_to_goal(user_input: str) -> str | None:
    kws = extract_keywords(user_input)
    scores = {gid: len(set(kws) & set(c.lower() for c in meta["key_concepts"]))
              for gid, meta in goal_metadata.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None

# KG-RAG retrieval
def query_neo4j_kg(user_input: str) -> str:
    kws = extract_keywords(user_input)
    if not kws:
        return "No KG context found."
    triples, seen = [], set()
    with driver.session() as session:
        for kw in kws:
            result = session.run(
                """
                MATCH (n)
                WHERE any(prop IN keys(n)
                          WHERE toLower(toString(n[prop]))
                                CONTAINS toLower($q))
                WITH n
                MATCH path=(n)-[r*1..2]-(m)
                RETURN DISTINCT n, m, r LIMIT 50
                """, {"q": kw})
            for rec in result:
                n, m, rels = rec["n"], rec["m"], rec["r"]
                a = n.get("name", n.get("title", "[Unnamed]"))
                b = m.get("name", m.get("title", "[Unnamed]"))
                at = list(n.labels)[0] if n.labels else "Node"
                bt = list(m.labels)[0] if m.labels else "Node"
                rt = ",".join({r.type for r in rels})
                line = f"{a} ({at}) —[{rt}]→ {b} ({bt})"
                if line not in seen:
                    seen.add(line)
                    triples.append(line)
    return "Knowledge Graph Context for '"+user_input+"':\n" + "\n".join(triples)

# Quiz engine
def dynamic_quiz(goal_id, lesson_content):
    prompt = (
        f"You are an occupational therapy tutor. Based only on this lesson, create a 3-question quiz:\n" +
        lesson_content +
        "\nLabel Q1, Q2, Q3. Return questions only."
    )
    quiz_text = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    questions = re.findall(r"(Q[1-3][:\)]\s*.+?)(?=(?:\nQ[1-3]|$))", quiz_text, flags=re.S)
    if len(questions) < 3:
        print("❌ Failed to generate quiz questions.")
        return
    correct = 0
    for q in questions:
        print(q)
        ans = input("Your answer: ").strip()
        eval_prompt = (
            f"Lesson:\n{lesson_content}\nQuestion:\n{q}\nAnswer:\n{ans}\n"
            "Reply 'yes' or 'no' and one sentence feedback."
        )
        fb = llm.invoke([HumanMessage(content=eval_prompt)]).content.strip()
        print("🧠 Feedback:", fb)
        if fb.lower().startswith("yes"):
            correct += 1
    print(f"✅ You answered {correct}/{len(questions)} correctly.")
    user_progress[goal_id] = "mastered" if correct >= 2 else "in_progress"
    save_progress(user_progress)

# Lesson generator
def tutor_lesson(goal_id):
    desc = goal_metadata[goal_id]["description"]
    prompt = f"You are a dementia tutor. Teach step by step: {desc}"
    lesson = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    print("📖 Lesson:\n", lesson)
    return lesson

# Session with KG-RAG
def run_goal_session(goal_id):
    lesson = tutor_lesson(goal_id)
    conversation = [HumanMessage(content=lesson)]
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == "quiz":
            dynamic_quiz(goal_id, lesson)
            break
        if user_input.lower() in {"exit", "menu"}:
            break
        kg = query_neo4j_kg(user_input)
        msgs = [SystemMessage(content=kg)] + conversation + [HumanMessage(content=user_input)]
        reply = llm.invoke(msgs).content.strip()
        print("Tutor:", reply)
        conversation.append(HumanMessage(content=user_input))
        conversation.append(HumanMessage(content=reply))
    print("Returning to menu...")

# CLI utilities
def list_goals():
    print("Available Learning Goals:")
    for k, meta in goal_metadata.items():
        status = user_progress.get(k)
        if status != "mastered":
            print(f"- {k}: {meta['description']} [{status}]")


def goal_selection_loop():
    while True:
        list_goals()
        cmd = input(">>> ").strip()
        if cmd.lower() == "exit":
            break
        if cmd.lower() == "progress":
            print(user_progress)
        elif cmd in goal_metadata and user_progress[cmd] != "mastered":
            run_goal_session(cmd)
        else:
            m = match_input_to_goal(cmd)
            if m and user_progress.get(m) != "mastered":
                run_goal_session(m)
            else:
                print("Invalid choice. Please try again.")

if __name__ == "__main__":
    print("🎓 Welcome to the AI Dementia Tutor!")
    goal_selection_loop()
