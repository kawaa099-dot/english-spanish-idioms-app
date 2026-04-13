import streamlit as st
import uuid

from utils import (
    load_idioms, generate_audio, init_db, add_favorite, get_favorite,
    detect_idioms, translate_literal, build_examples_map, remove_favorite,
    generate_ai_question_dynamic, update_analytics, get_learning_stats
)

st.set_page_config(page_title="Idioms Learning App", layout="wide")
st.title("English–Spanish Idioms Learning App")

# Load resources
idiom_map = load_idioms("idioms.json")
idioms = sorted(idiom_map.keys())
conn = init_db()
examples_map = build_examples_map()

# Sidebar
mode = st.sidebar.selectbox(
    "Choose a mode",
    ["Explore Idioms", "Idioms in sentences", "Quiz time!", "Favorites", "Learning Analytics"]
)

# EXPLORE

if mode == "Explore Idioms":
    search = st.text_input("Search idiom:")
    filtered = [i for i in idioms if search.lower() in i.lower()]
    selected = st.selectbox("Choose idiom:", filtered)

    if selected:
        st.subheader("Natural Spanish Meaning")
        st.write(idiom_map[selected])

        st.subheader("Literal Translation")
        st.write(translate_literal(selected))

        st.subheader("Examples:")
        examples = examples_map.get(selected.lower(), [])
        if examples:
            for ex in examples[:3]:
                st.write("**English:**", ex["en"])
                st.write("**Spanish:**", ex["es"])
                st.write("---")
        else:
            st.write("No examples found.")

        st.subheader("Audio")
        audio_file = generate_audio(selected)
        st.audio(audio_file)

        if st.button("⭐ Add to Favorites"):
            add_favorite(conn, selected)
            st.success("Added to favorites!")


# DETECT 

elif mode == "Idioms in sentences":
    text = st.text_area("Write your text:")

    if st.button("Detect"):
        found = detect_idioms(text, idioms)
        
        if not found:
            st.info("No idioms detected.")
        else:
            # Highlight detected idioms in the text
            highlighted_text = text
            for idiom in found:
                highlighted_text = highlighted_text.replace(
                    idiom,
                    f"<span style='background-color: #ffff00; font-weight:bold'>{idiom}</span>"
                )
            st.markdown("### Your Text with Detected Idioms")
            st.markdown(highlighted_text, unsafe_allow_html=True)
            st.markdown("---")

            # Show idiom details
            for idiom in found:
                with st.expander(f"🔹 {idiom}"):
                    st.write("**Meaning:**", idiom_map[idiom])
                    st.write("**Literal Translation:**", translate_literal(idiom))

                    # Examples
                    examples = examples_map.get(idiom.lower(), [])
                    if examples:
                        st.subheader("Examples:")
                        for ex in examples[:2]:
                            st.write("•", ex["en"])
                    else:
                        st.write("No examples available.")

                    # Audio
                    audio_file = generate_audio(idiom)
                    st.audio(audio_file)

                    # Practice button
                    if st.button(f"Practice '{idiom}'"):
                        st.session_state.quiz = generate_ai_question_dynamic([idiom])
                        st.rerun()

# QUIZ

elif mode == "Quiz time!":

    st.header("🎮Quiz")

    if "xp" not in st.session_state:
        st.session_state.xp = 0

    if "used_questions" not in st.session_state:
        st.session_state.used_questions = set()


    st.metric("XP", st.session_state.xp)

    if "quiz" not in st.session_state:
        st.session_state.quiz = generate_ai_question_dynamic(idioms,examples_map,st.session_state.used_questions)

    if "question_id" not in st.session_state:
        st.session_state.question_id = str(uuid.uuid4())

    quiz = st.session_state.quiz

    st.write("Fill in the blank")
    st.write(quiz["question"])

    user_answer = st.radio(
        "Choose your answer:",
        quiz["options"],
        key=st.session_state.question_id  
    )

    if st.button("Submit"):

        correct = user_answer == quiz["answer"]
        update_analytics(conn, quiz["answer"], correct)

        if correct:
            st.success("Correct! +5 XP")
            st.session_state.xp += 5
        else:
            st.error(f"Wrong! Correct answer: {quiz['answer']}")

    if st.button("New Question"):
        st.session_state.quiz = generate_ai_question_dynamic(idioms, examples_map,used_questions)
        st.session_state.question_id = str(uuid.uuid4()) 
        st.rerun()

# ANALYTICS

elif mode == "Learning Analytics":

    st.header("Learning Dashboard")

    stats = get_learning_stats(conn)

    if not stats:
        st.info("No quiz data yet. Try the quiz first!")
    else:
        import numpy as np
        import pandas as pd

        total_idioms = len(idioms)

        mastered = len([i for i,a,c,acc in stats if acc >= 0.8 and a >= 2])
        practiced = len([i for i,a,c,acc in stats if a > 0])
        weak = len([i for i,a,c,acc in stats if acc < 0.6 and a > 0])

        avg_acc = np.mean([acc for _,a,_,acc in stats if a > 0]) if stats else 0

        # XP system
        xp = st.session_state.get("xp",0)
        next_level = 100
        xp_progress = xp / next_level

        st.subheader("⭐ XP Progress")

        st.metric("Total XP", xp)
        st.progress(xp_progress)

        st.markdown("---")

        # Global Stats
        st.subheader("🏆 Your Progress")

        col1,col2,col3,col4 = st.columns(4)

        with col1:
            st.metric(
                label="🎯 Mastered",
                value=f"{mastered}/{total_idioms}",
            )
            with st.expander("❓ What is Mastered?"):
                st.caption("Idioms you have answered correctly at least 80% of the time.")
            st.progress(mastered/total_idioms)
        
        
        with col2:
            st.metric(
                label="📚 Practiced",
                value=f"{practiced}/{total_idioms}",
            )
            with st.expander("❓ What is Practiced?"):
                st.caption("Idioms you have attempted at least once.")
            st.progress(practiced/total_idioms)
        
        
        with col3:
            st.metric(
                label="⚠️ Weak",
                value=weak,
            )
            with st.expander("❓ What is Weak?"):
                st.caption("Idioms with accuracy below 60%.")
            st.progress(weak/total_idioms)
        
        
        with col4:
            st.metric(
                label="🎯 Accuracy",
                value=f"{avg_acc*100:.1f}%",
            )
            with st.expander("❓ What is Accuracy?"):
                st.caption("Your average correct rate across all attempted idioms.")
            st.progress(avg_acc)        

        st.markdown("---")

        # Performance Chart
        st.subheader("📊 Performance")

        df = pd.DataFrame(stats, columns=["Idiom","Attempts","Correct","Accuracy"])

        df_sorted = df.sort_values("Accuracy")

        st.bar_chart(
            df_sorted.set_index("Idiom")["Accuracy"]
        )

        st.markdown("---")

        # Weak Idioms
        st.subheader("🔥 Idioms You Should Practice")

        weak_list = df[df["Accuracy"] < 0.6]["Idiom"].tolist()

        if weak_list:
            for w in weak_list:
                st.warning(f"Practice: **{w}**")
        else:
            st.success("Great job! No weak idioms 🎉")

        st.markdown("---")

        # Full Table
        st.subheader("Detailed Stats")

        st.dataframe(df)
# FAVORITES

elif mode == "Favorites":

    st.header("Your Favorites")
    favs = get_favorite(conn)

    if favs:
        for idiom in favs:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write("⭐", idiom)
                st.write("Meaning:", idiom_map.get(idiom, "Unknown"))
            with col2:
                if st.button("Remove", key=f"remove_{idiom}"):
                    remove_favorite(conn, idiom)
                    st.rerun()
            st.write("---")
    else:
        st.write("No favorites yet.")