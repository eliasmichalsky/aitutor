# kgrag.py
from typing import Annotated, List
from typing_extensions import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from neo4j import GraphDatabase
import os, re, json

# Ladda miljövariabler
dotenv_loaded = load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Initiera Neo4j-driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
try:
    with driver.session() as session:
        msg = session.run("RETURN 'Neo4j ansluten' AS message").single()["message"]
        print("[✅ Neo4j]", msg)
except Exception as e:
    print("[❌ Neo4j FEL] Kunde inte ansluta:", e)

# Fil för att spara användarens framsteg samt initiera LLM
PROGRESS_FILE = "user_progress.json"
llm = ChatOpenAI(model="openai/o3-mini")

# Typ för sessionstillstånd\ nclass State(TypedDict):
class State(TypedDict):
    messages: Annotated[List[HumanMessage], add_messages]
    user_query: str


# Ladda och spara framsteg
def load_progress() -> dict:
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {f"LG{i}": "inte påbörjad" for i in range(1, 11)}

def save_progress(progress: dict):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)

user_progress = load_progress()

# Metadata för inlärningsmål
goal_metadata = {
    "LG1": {"description": "Utför strukturerade kognitiva bedömningar med MoCA och CID.", "key_concepts": ["MoCA", "CID", "kognitiv funktion", "vardagsliv"]},
    "LG2": {"description": "Tolka screeningresultat och koppla till påverkan på dagliga aktiviteter och delaktighet.", "key_concepts": ["screeningresultat", "daglig aktivitet", "kognitiva brister", "delaktighet"]},
    "LG3": {"description": "Genomför miljöbedömningar för att identifiera stödjande eller hindrande faktorer.", "key_concepts": ["miljöfaktorer", "funktion", "säkerhet", "delaktighet"]},
    "LG4": {"description": "Anpassa miljön för att stödja självständighet, säkerhet och orientering.", "key_concepts": ["anpassningar", "miljö", "självständighet", "orientering", "säkerhet"]},
    "LG5": {"description": "Rekommendera och introducera kognitiva hjälpmedel anpassade efter individens behov.", "key_concepts": ["hjälpmedel", "kognitivt stöd", "demensstadier", "behovsanalys"]},
    "LG6": {"description": "Planera och genomföra aktivitetsbaserade interventioner för självständighet och meningsfullhet.", "key_concepts": ["interventioner", "aktivitetsbaserat", "mening", "självständighet"]},
    "LG7": {"description": "Vägled anhöriga och personal i strategier för vardagsaktiviteter med personcentrerat fokus.", "key_concepts": ["anhörigstrategier", "personcentrerad vård", "anhöriga", "personal"]},
    "LG8": {"description": "Dokumentera arbetsterapiinsatser enligt ICF-ramverket.", "key_concepts": ["dokumentation", "ICF", "insats", "kommunikation"]},
    "LG9": {"description": "Skriv strukturerade remissvar baserade på funktionella bedömningar.", "key_concepts": ["remiss", "bedömning", "rapport", "demensutredning"]},
    "LG10": {"description": "Anpassa insatser utifrån sjukdomens progression från lindrad svikt till svår demens.", "key_concepts": ["insats", "progression", "lindrad svikt", "svår demens"]}
}

# Extrahera nyckelord från användarinput
def extract_keywords(user_input: str) -> List[str]:
    stopwords = {"vad","är","berätta","om","och","kan","du","jag","vill","lära"}
    tokens = re.findall(r"\b\w+\b", user_input.lower())
    keywords = [w for w in tokens if w not in stopwords and len(w) > 2]
    return keywords if keywords else tokens

# Matcha input till relevant mål
def match_input_to_goal(user_input: str) -> str | None:
    kws = extract_keywords(user_input)
    scores = {gid: len(set(kws) & set(c.lower() for c in meta["key_concepts"])) for gid, meta in goal_metadata.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None

# Hämta kontext från kunskapsgrafen
def query_neo4j_kg(user_input: str) -> str:
    kws = extract_keywords(user_input)
    if not kws:
        return "Ingen KG-kontekst hittad."
    triples, seen = [], set()
    with driver.session() as session:
        for kw in kws:
            result = session.run(
                """
                MATCH (n)
                WHERE any(prop IN keys(n) WHERE toLower(toString(n[prop])) CONTAINS toLower($q))
                WITH n
                MATCH path=(n)-[r*1..2]-(m)
                RETURN DISTINCT n, m, r LIMIT 50
                """, {"q": kw})
            for rec in result:
                n, m, rels = rec["n"], rec["m"], rec["r"]
                a = n.get("name", n.get("title", "[Namnlös]"))
                b = m.get("name", m.get("title", "[Namnlös]"))
                rt = ",".join({r.type for r in rels})
                line = f"{a} —[{rt}]→ {b}"
                if line not in seen:
                    seen.add(line)
                    triples.append(line)
    return "KG-kontekst för '" + user_input + "':\n" + "\n".join(triples)

# Skapa quizfrågor som en lista (för bot_interface.py)
def dynamic_quiz_list(goal_id: str, lesson_text: str) -> list[str]:
    prompt = (
        f"Du är en arbetsterapeutisk handledare. Skapa 3 quizfrågor baserat på lektionen:\n"
        f"{lesson_text}\n\n"
        f"Märk frågorna som Q1, Q2, Q3. Returnera endast frågorna."
    )
    quiz_text = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    questions = re.findall(r"(Q[1-3][:\)]\s*.+?)(?=(?:\nQ[1-3]|$))", quiz_text, flags=re.S)
    return questions


# Utvärdera elevens svar
def evaluate_answer(answer: str, question: str, lesson_text: str) -> str:
    prompt = (
        f"Du utvärderar ett svar på svenska.\n\n"
        f"Lektion:\n{lesson_text}\n\n"
        f"Fråga:\n{question}\n\n"
        f"Svar:\n{answer}\n\n"
        "Om det visar förståelse, svara 'ja' följt av en uppmuntrande mening.\n"
        "Om inte, svara 'nej' och en kort mening om vad som saknas."
    )
    return llm.invoke([HumanMessage(content=prompt)]).content.strip()


# Generera lektion
def tutor_lesson(goal_id, level="Nybörjare"):
    desc = goal_metadata[goal_id]["description"]
    prompt = f"Du är en handledare i demens. Förklara detta steg-för-steg på {'enkel' if level == 'Nybörjare' else 'medel' if level == 'Medel' else 'avancerad'} nivå:\n\n{desc}"
    lesson = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    print("📖 Lektion:\n", lesson)
    return lesson


# Session med KG-RAG
def run_goal_session(goal_id: str):
    lesson = tutor_lesson(goal_id)
    conversation = [HumanMessage(content=lesson)]
    while True:
        user_input = input("\nDu: ").strip()
        if user_input.lower() == "quiz":
            questions = dynamic_quiz_list(goal_id, lesson)
            print("Quiz-frågor:")
            for q in questions:
                print(q)
            break
        if user_input.lower() in {"avsluta", "meny"}:
            break
        kg = query_neo4j_kg(user_input)
        msgs = [SystemMessage(content=kg)] + conversation + [HumanMessage(content=user_input)]
        reply = llm.invoke(msgs).content.strip()
        print("Tutor:", reply)
        conversation.extend([HumanMessage(content=user_input), HumanMessage(content=reply)])
    print("Återgår till menyn...")

# CLI-verktyg
def list_goals():
    print("Tillgängliga inlärningsmål:")
    for k, meta in goal_metadata.items():
        status = user_progress.get(k)
        if status != "mastered":
            print(f"- {k}: {meta['description']} [{status}]")


def goal_selection_loop():
    progress = load_progress()
    while True:
        list_goals()
        cmd = input(">>> ").strip()
        if cmd.lower() == "avsluta":
            break
        if cmd.lower() == "framsteg":
            print(progress)
        elif cmd in goal_metadata and progress.get(cmd) != "mastered":
            run_goal_session(cmd)
            progress = load_progress()
        else:
            m = match_input_to_goal(cmd)
            if m and progress.get(m) != "mastered":
                run_goal_session(m)
                progress = load_progress()
            else:
                print("Ogiltigt val. Försök igen.")

if __name__ == "__main__":
    print("🎓 Välkommen till AI Demenstutör!")
    goal_selection_loop()
