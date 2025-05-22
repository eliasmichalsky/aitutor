# bot_interface.py
import os, re
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage

import kgrag
from kgrag import llm, goal_metadata, load_progress, save_progress

# Ladda miljövariabler
load_dotenv()

# Konfigurera sidan
st.set_page_config(page_title="Demenstutör", layout="wide")

# Välj kunskapsnivå
st.sidebar.markdown("<h3 style='color:green'>🎚️ Välj din nivå</h3>", unsafe_allow_html=True)
level = st.sidebar.radio("Vilken kunskapsnivå har du?", ["Nybörjare", "Medel", "Expert"])

# Ladda framsteg
progress = load_progress()

# Förhindra byte av mål om quiz är pågående
goal_data = st.session_state.get("goal_data", {})
selected_goal_data = goal_data.get(st.session_state.get("selected_goal"), {})

quiz_qs = selected_goal_data.get("quiz_qs", [])
step = selected_goal_data.get("step", 0)

quiz_locked = len(quiz_qs) > 0 and step < len(quiz_qs)


# Målval
st.sidebar.markdown("<h3 style='color:green'>📋 Välj inlärningsmål</h3>", unsafe_allow_html=True)
if "selected_goal" not in st.session_state:
    st.session_state.selected_goal = None
if "goal_data" not in st.session_state:
    st.session_state.goal_data = {}

# Hantera målbyte och initialisering
for gid, meta in goal_metadata.items():
    label = f"✅ {meta['description']}" if progress.get(gid) == "mastered" else meta['description']
    if st.sidebar.button(label, key=gid, disabled=quiz_locked):
        st.session_state.selected_goal = gid

# Om inget mål valt, välj första som inte är klart
if st.session_state.selected_goal is None:
    for gid, status in progress.items():
        if status != "mastered":
            st.session_state.selected_goal = gid
            break

selected_goal = st.session_state.selected_goal
selected_desc = goal_metadata[selected_goal]["description"]

# Initiera datamall per mål
if selected_goal not in st.session_state.goal_data:
    st.session_state.goal_data[selected_goal] = {
        "lesson": None,
        "quiz_qs": [],
        "step": 0,
        "correct": 0,
        "chat_history": []
    }

# Använd aktuell målspecifik data
goal_state = st.session_state.goal_data[selected_goal]

# Rubrik
st.title("🧠 AI Demenstutör")
st.header(f"🎯 {selected_desc}")

# Cache-funktioner
@st.cache_resource(ttl=3600)
def generate_lesson_cached(goal_id: str, level: str) -> str:
    return kgrag.tutor_lesson(goal_id, level)

@st.cache_data(ttl=3600)
def generate_quiz_cached(goal_id: str, lesson_text: str) -> list[str]:
    return kgrag.dynamic_quiz_list(goal_id, lesson_text)

@st.cache_data(ttl=3600)
def evaluate_answer_cached(answer: str, question: str, lesson_text: str) -> str:
    return kgrag.evaluate_answer(answer, question, lesson_text)

# Visa lektion
if st.button("📖 Visa lektion"):
    goal_state["lesson"] = generate_lesson_cached(selected_goal, level)

# Lektionstext
if goal_state["lesson"]:
    st.subheader("Lektion")
    st.markdown(f"<div style='font-size:18px;line-height:1.6;'>{goal_state['lesson']}</div>", unsafe_allow_html=True)

# Chatt
st.subheader("Chatt")
for role, msg in goal_state["chat_history"]:
    st.markdown(f"**{role}:** {msg}")

def handle_user_input():
    user_q = st.session_state.get("chat_input", "").strip()
    if not user_q:
        st.warning("Skriv en fråga innan du skickar.")
        return

    kg_ctx = kgrag.query_neo4j_kg(user_q)

    msgs = [
        SystemMessage(content="Du är en tydlig och vänlig handledare inom demens. Svara på svenska, enkelt och pedagogiskt."),
        SystemMessage(content=f"Lektion:\n{goal_state['lesson']}"),
        SystemMessage(content=f"Kunskap från graf:\n{kg_ctx}")
    ]

    for role, msg in goal_state["chat_history"]:
        if role == "Du":
            msgs.append(HumanMessage(content=msg))
        else:
            msgs.append(SystemMessage(content=msg))

    msgs.append(HumanMessage(content=user_q))

    with st.spinner("Tänker..."):
        resp = llm.invoke(msgs).content.strip()

    goal_state["chat_history"].append(("Du", user_q))
    goal_state["chat_history"].append(("Tutor", resp))

# Chattfält
st.text_input("Ställ en fråga till tutorn:", key="chat_input", on_change=handle_user_input)

# Quiz
if goal_state["lesson"]:
    st.subheader("Quiz")
    if st.button("📝 Starta quiz"):
        goal_state["quiz_qs"] = generate_quiz_cached(selected_goal, goal_state["lesson"])
        goal_state["step"] = 0
        goal_state["correct"] = 0

    if goal_state["quiz_qs"]:
        idx = goal_state["step"]
        total = len(goal_state["quiz_qs"])
        if idx < total:
            raw = goal_state["quiz_qs"][idx]
            text = re.sub(r"^Q[0-9]+[:\)]\s*", "", raw)
            st.markdown(f"**Fråga {idx+1}/{total}:** {text}")
            ans = st.text_input("Svar:", key=f"ans_{idx}")
            if st.button("Skicka svar", key=f"sub_{idx}") and ans:
                fb = evaluate_answer_cached(ans, raw, goal_state["lesson"])
                if fb.lower().startswith("ja"):
                    goal_state["correct"] += 1
                goal_state["step"] += 1
                st.markdown(f"🧠 {fb}")
        else:
            score = goal_state["correct"]
            st.success(f"Du fick {score}/{total} rätt.")
            progress[selected_goal] = "mastered" if score >= 2 else "in_progress"
            save_progress(progress)
            if score >= 2:
                st.balloons()
                st.success("Mål uppnått!")
            else:
                st.info("Fortsätt träna på detta mål.")
