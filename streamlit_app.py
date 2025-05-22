# streamlit_app.py
import os
import re
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage

# Import from our integrated backend
import kgrag

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="AI Demenstutör", 
    layout="wide",
    page_icon="🧠"
)

# Initialize session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "selected_goal" not in st.session_state:
    st.session_state.selected_goal = None

if "goal_data" not in st.session_state:
    st.session_state.goal_data = {}

# Sidebar - Knowledge level selection
st.sidebar.markdown("<h3 style='color:green'>🎚️ Välj din nivå</h3>", unsafe_allow_html=True)
level = st.sidebar.radio("Vilken kunskapsnivå har du?", ["Nybörjare", "Medel", "Expert"])

# Load progress
progress = kgrag.load_progress()

# Check if quiz is in progress to lock goal switching
goal_data = st.session_state.get("goal_data", {})
selected_goal_data = goal_data.get(st.session_state.get("selected_goal"), {})
quiz_qs = selected_goal_data.get("quiz_qs", [])
step = selected_goal_data.get("step", 0)
quiz_locked = len(quiz_qs) > 0 and step < len(quiz_qs)

# Sidebar - Learning goal selection
st.sidebar.markdown("<h3 style='color:green'>📋 Välj inlärningsmål</h3>", unsafe_allow_html=True)

# Handle goal selection and initialization
for gid, meta in kgrag.goal_metadata.items():
    label = f"✅ {meta['description']}" if progress.get(gid) == "mastered" else meta['description']
    if st.sidebar.button(label, key=gid, disabled=quiz_locked):
        st.session_state.selected_goal = gid
        # Reset conversation when switching goals
        st.session_state.conversation_history = []

# If no goal selected, select first incomplete goal
if st.session_state.selected_goal is None:
    for gid, status in progress.items():
        if status != "mastered":
            st.session_state.selected_goal = gid
            break

selected_goal = st.session_state.selected_goal
selected_desc = kgrag.goal_metadata[selected_goal]["description"]

# Initialize goal-specific data
if selected_goal not in st.session_state.goal_data:
    st.session_state.goal_data[selected_goal] = {
        "lesson": None,
        "quiz_qs": [],
        "step": 0,
        "correct": 0,
        "chat_history": []
    }

goal_state = st.session_state.goal_data[selected_goal]

# Main interface
st.title("🧠 AI Demenstutör")
st.header(f"🎯 {selected_desc}")

# Progress indicator
col1, col2, col3 = st.columns(3)
with col1:
    total_goals = len(kgrag.goal_metadata)
    mastered_goals = sum(1 for status in progress.values() if status == "mastered")
    st.metric("Uppnådda mål", f"{mastered_goals}/{total_goals}")

with col2:
    current_status = progress.get(selected_goal, "inte påbörjad")
    st.metric("Aktuellt mål status", current_status)

with col3:
    if goal_state["quiz_qs"]:
        quiz_progress = f"{goal_state['step']}/{len(goal_state['quiz_qs'])}"
        st.metric("Quiz framsteg", quiz_progress)

# Cache functions for performance
@st.cache_data(ttl=3600)
def generate_lesson_cached(goal_id: str, level: str) -> str:
    return kgrag.tutor_lesson(goal_id, level)

@st.cache_data(ttl=3600)
def generate_quiz_cached(goal_id: str, lesson_text: str) -> list[str]:
    return kgrag.dynamic_quiz_list(goal_id, lesson_text)

@st.cache_data(ttl=3600)
def evaluate_answer_cached(answer: str, question: str, lesson_text: str) -> str:
    return kgrag.evaluate_answer(answer, question, lesson_text)

# Lesson section
st.subheader("📖 Lektion")
col1, col2 = st.columns([1, 4])

with col1:
    if st.button("Visa lektion", type="primary"):
        with st.spinner("Genererar lektion..."):
            goal_state["lesson"] = generate_lesson_cached(selected_goal, level)

with col2:
    if st.button("Ny lektion", help="Generera en ny version av lektionen"):
        # Clear cache and generate new lesson
        st.cache_data.clear()
        with st.spinner("Genererar ny lektion..."):
            goal_state["lesson"] = kgrag.tutor_lesson(selected_goal, level)

# Display lesson content
if goal_state["lesson"]:
    with st.expander("Lektionsinnehåll", expanded=True):
        st.markdown(f"<div style='font-size:18px;line-height:1.6;'>{goal_state['lesson']}</div>", 
                   unsafe_allow_html=True)

# Chat section
st.subheader("💬 Chatt med AI-tutorn")

# Display chat history
chat_container = st.container()
with chat_container:
    for role, msg in goal_state["chat_history"]:
        if role == "Du":
            st.chat_message("user").write(msg)
        else:
            st.chat_message("assistant").write(msg)

# Chat input
def handle_chat_input():
    user_input = st.session_state.get("chat_input", "").strip()
    if not user_input:
        st.warning("Skriv en fråga innan du skickar.")
        return

    # Add user message to chat history
    goal_state["chat_history"].append(("Du", user_input))
    
    # Process through KG-RAG system
    with st.spinner("Tänker..."):
        response = kgrag.stream_graph_updates(user_input, st.session_state.conversation_history.copy())
    
    # Add only the assistant response to chat history (KG context is handled internally)
    if response:
        goal_state["chat_history"].append(("Tutor", response))

# Chat input field
chat_input = st.chat_input("Ställ en fråga till tutorn...")
if chat_input:
    st.session_state.chat_input = chat_input
    handle_chat_input()
    st.rerun()

# Quiz section
if goal_state["lesson"]:
    st.subheader("📝 Quiz")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("Starta quiz", disabled=bool(goal_state["quiz_qs"])):
            with st.spinner("Genererar quiz..."):
                goal_state["quiz_qs"] = generate_quiz_cached(selected_goal, goal_state["lesson"])
                goal_state["step"] = 0
                goal_state["correct"] = 0
            st.rerun()
    
    with col2:
        if goal_state["quiz_qs"] and st.button("Återställ quiz"):
            goal_state["quiz_qs"] = []
            goal_state["step"] = 0
            goal_state["correct"] = 0
            st.rerun()

    # Display quiz questions
    if goal_state["quiz_qs"]:
        idx = goal_state["step"]
        total = len(goal_state["quiz_qs"])
        
        if idx < total:
            # Current question
            raw_question = goal_state["quiz_qs"][idx]
            clean_question = re.sub(r"^Q[0-9]+[:\)]\s*", "", raw_question)
            
            st.markdown(f"**Fråga {idx+1}/{total}:**")
            st.markdown(f"*{clean_question}*")
            
            # Answer input
            answer_key = f"quiz_answer_{idx}"
            user_answer = st.text_area("Ditt svar:", key=answer_key, height=100)
            
            if st.button("Skicka svar", key=f"submit_{idx}") and user_answer.strip():
                with st.spinner("Utvärderar svar..."):
                    feedback = evaluate_answer_cached(user_answer, raw_question, goal_state["lesson"])
                
                # Check if answer is correct
                if feedback.lower().startswith("ja"):
                    goal_state["correct"] += 1
                    st.success(f"✅ {feedback}")
                else:
                    st.error(f"❌ {feedback}")
                
                # Move to next question
                goal_state["step"] += 1
                
                # Auto-advance after a short delay
                st.balloons() if feedback.lower().startswith("ja") else None
                st.rerun()
        
        else:
            # Quiz completed
            score = goal_state["correct"]
            percentage = (score / total) * 100
            
            st.subheader("🎉 Quiz slutförd!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Resultat", f"{score}/{total}")
            with col2:
                st.metric("Procent", f"{percentage:.0f}%")
            with col3:
                if score >= 2:  # Passing threshold
                    st.success("✅ Godkänt!")
                else:
                    st.error("❌ Ej godkänt")
            
            # Update progress
            if score >= 2:
                progress[selected_goal] = "mastered"
                st.balloons()
                st.success("🎯 Inlärningsmål uppnått!")
            else:
                progress[selected_goal] = "in_progress"
                st.info("📚 Fortsätt träna på detta mål.")
            
            # Save progress
            kgrag.save_progress(progress)
            
            # Option to retake quiz
            if st.button("Gör om quiz"):
                goal_state["quiz_qs"] = []
                goal_state["step"] = 0
                goal_state["correct"] = 0
                st.rerun()

# Sidebar - Additional features
st.sidebar.markdown("---")
st.sidebar.markdown("<h3 style='color:green'>📊 Framsteg</h3>", unsafe_allow_html=True)

# Progress overview
for gid, meta in kgrag.goal_metadata.items():
    status = progress.get(gid, "inte påbörjad")
    emoji = "✅" if status == "mastered" else "🔄" if status == "in_progress" else "⭕"
    st.sidebar.markdown(f"{emoji} {gid}: {status}")

# Export/Import progress
st.sidebar.markdown("---")
if st.sidebar.button("Exportera framsteg"):
    st.sidebar.download_button(
        "Ladda ner framsteg.json",
        data=str(progress),
        file_name="progress.json",
        mime="application/json"
    )

# System information
with st.sidebar.expander("ℹ️ Systeminformation"):
    st.write(f"**Vald nivå:** {level}")
    st.write(f"**Aktivt mål:** {selected_goal}")
    st.write(f"**Chatthistorik:** {len(goal_state['chat_history'])} meddelanden")
    st.write(f"**KG-anslutning:** {'✅ Ansluten' if kgrag.driver else '❌ Frånkopplad'}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    🧠 AI Demenstutör - Använder Knowledge Graph-Enhanced RAG för personaliserad inlärning
    </div>
    """, 
    unsafe_allow_html=True
)