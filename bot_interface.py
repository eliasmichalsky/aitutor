from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os, json, re
import streamlit as st

# Load environment variables
load_dotenv()

# Page config must be the first Streamlit call
st.set_page_config(page_title="Dementia Tutor", layout="wide")

# Cache the OpenAI client so it's instantiated only once
@st.cache_resource
def get_llm():
    return ChatOpenAI(model="openai/o3-mini")
llm = get_llm()

# Cache lesson generation to speed up repeated calls\@st.cache_data(ttl=3600)
def generate_lesson_cached(goal_id: str, language: str, level: str, desc: str) -> str:
    lang_prompt = "på svenska" if language == "Svenska" else "in clear English"
    prompt = f"You are a dementia tutor. Teach this concept step by step {lang_prompt} for a {level.lower()} learner:\n{desc}"
    return llm.invoke([HumanMessage(content=prompt)]).content.strip()

# Cache quiz generation to speed up repeated calls\@st.cache_data(ttl=3600)
def generate_quiz_cached(lesson_text: str, language: str, level: str) -> list[str]:
    lang_prompt = "på svenska" if language == "Svenska" else "in English"
    prompt = (
        f"You are an OT tutor. Based on the lesson below, create a 3-question quiz {lang_prompt}"
        f" for a {level.lower()} learner:\n\n{lesson_text}\n\nLabel Q1, Q2, Q3. Return questions only."
    )
    quiz_text = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    pattern = r"(Q[0-9]+[:\)]\s*.+?)(?=(?:\nQ[0-9]+[:\)]|\Z))"
    return re.findall(pattern, quiz_text, flags=re.S)

# Evaluate answers without caching
def evaluate_answer(answer: str, question: str, lesson_text: str, language: str, level: str) -> str:
    lang_prompt = "på svenska" if language == "Svenska" else "in English"
    prompt = (
        f"You are evaluating an answer {lang_prompt} at a {level.lower()} level."
        f"\n\nLesson:\n{lesson_text}\n\nQuestion:\n{question}\n\nAnswer:\n{answer}\n\n"
        "If it shows understanding, reply 'yes' and one encouraging sentence. "
        "If not, reply 'no' and one short sentence on what's missing."
    )
    return llm.invoke([HumanMessage(content=prompt)]).content.strip()

# Learning goal metadata
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

PROGRESS_FILE = "user_progress.json"

def load_progress() -> dict:
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {gid: "not_started" for gid in goal_metadata}

def save_progress(progress: dict):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)

# Streamlit App Layout
# (page config moved to top, see above)
st.title("🧠 AI Dementia Tutor")

# Sidebar: Settings
st.sidebar.markdown("<h3 style='color:green'>🌐 Settings</h3>", unsafe_allow_html=True)
language = st.sidebar.selectbox("Language / Språk", ["English", "Svenska"])
level = st.sidebar.selectbox("Your level", ["Beginner", "Intermediate", "Advanced"])

# Sidebar: Learning Goals
progress = load_progress()
st.sidebar.markdown("<h3 style='color:green'>📋 Select Learning Goal</h3>", unsafe_allow_html=True)
if 'selected_goal' not in st.session_state:
    st.session_state.selected_goal = None
for gid, meta in goal_metadata.items():
    label = f"✅ {meta['description']}" if progress.get(gid)=="mastered" else meta['description']
    if st.sidebar.button(label, key=gid):
        st.session_state.selected_goal = gid
# default to first non-mastered
if st.session_state.selected_goal is None:
    for gid, stt in progress.items():
        if stt != "mastered": st.session_state.selected_goal = gid; break
selected_goal = st.session_state.selected_goal
selected_desc = goal_metadata[selected_goal]['description']

# Session state defaults
defaults = {'lesson': "", 'quiz_qs': [], 'step': 0, 'correct': 0, 'feedback': "", 'chat_history': []}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Reset when goal changes
if st.session_state.get('last_goal') != selected_goal:
    for k in defaults:
        st.session_state[k] = defaults[k]
    st.session_state['last_goal'] = selected_goal

# Main content
st.header(f"🎯 {selected_desc}")

# Show Lesson
if st.button("📖 Show Lesson"):
    st.session_state.lesson = generate_lesson_cached(selected_goal, language, level, selected_desc)

# Display Lesson and Chat
if st.session_state.lesson:
    st.subheader("Lesson")
    st.markdown(
        f"<div style='font-size:18px;line-height:1.6;'>{st.session_state.lesson}</div>",
        unsafe_allow_html=True
    )

    # 1) render chat history
    for role, msg in st.session_state.chat_history:
        st.markdown(f"**{role}:** {msg}")

    # 2) then your form at the bottom:
    with st.form(key="chat_form", clear_on_submit=True):
        user_q = st.text_input("Ask the Tutor:", placeholder="Type your question...")
        send = st.form_submit_button("Send")
        if send:
            handle_send(user_q)


    # Chat input and send button at bottom
    def handle_send():
        user_q = st.session_state.chat_input.strip()
        if user_q:
            with st.spinner("Thinking…"):
                msgs = [SystemMessage(content=f"Tutor on: {selected_desc} ({language}, {level})")]
                for _, past in st.session_state.chat_history:
                    msgs.append(HumanMessage(content=past))
                msgs.append(HumanMessage(content=user_q))
                resp = llm.invoke(msgs).content.strip()
            st.session_state.chat_history.append(("You", user_q))
            st.session_state.chat_history.append(("Tutor", resp))
        else:
            st.warning("Please enter a question.")

    st.text_input("Ask the Tutor:", key="chat_input", placeholder="Type your question...")
    st.button("Send", on_click=handle_send)

# Quiz section
st.subheader("Quiz")
if st.session_state.lesson and st.button("📝 Start Quiz"):
    st.session_state.quiz_qs = generate_quiz_cached(st.session_state.lesson, language, level)
    st.session_state.step = 0
    st.session_state.correct = 0
    st.session_state.feedback = ""

if st.session_state.quiz_qs:
    idx = st.session_state.step
    total = len(st.session_state.quiz_qs)
    if idx < total:
        raw = st.session_state.quiz_qs[idx]
        text = re.sub(r"^Q[0-9]+[:\)]\s*", "", raw)
        st.markdown(f"**Question {idx+1}/{total}:** {text}")
        ans = st.text_input("Answer:", key=f"ans_{idx}")
        if st.button("Submit", key=f"sub_{idx}"):
            if ans:
                fb = evaluate_answer(ans, raw, st.session_state.lesson, language, level)
                st.session_state.feedback = fb
                if fb.lower().startswith("yes"): st.session_state.correct += 1
                st.session_state.step += 1
            else:
                st.warning("Please enter an answer.")
        if st.session_state.feedback:
            st.markdown(f"🧠 {st.session_state.feedback}")
    else:
        score = st.session_state.correct
        st.success(f"You got {score}/{total} correct.")
        if score >= 2:
            progress[selected_goal] = "mastered"
            st.balloons(); st.success("Goal mastered!")
        else:
            progress[selected_goal] = "in_progress"; st.info("Keep working on this goal.")
        save_progress(progress)
