import streamlit as st
import torch
import re
import pandas as pd
from sentence_transformers import SentenceTransformer
from torch.nn.functional import normalize

# Cache Model for faster performance
@st.cache_resource
def load_model():
    return SentenceTransformer("all-mpnet-base-v2")

model = load_model()

# Mystery sentence (Hidden)
MYSTERY_SENTENCE = "A seed does not become a tree overnight; success need patience and care bring growth."

# Leaderboard Storage
if "leaderboard" not in st.session_state:
    st.session_state.leaderboard = []

# Step 1: Get Username
st.title("🔑 Unlock the AI's mystery text!")

if "username" not in st.session_state:
    st.subheader("Enter Your Team Name")
    username = st.text_input("Username:")
    
    if username:
        st.session_state.username = username
        st.rerun()

# Step 2: Passcode Entry
if "username" in st.session_state and "authenticated" not in st.session_state:
    st.subheader(f"👤 Welcome, {st.session_state.username}!")
    passcode = st.text_input("Enter Passcode:", type="password")

    if passcode == "Mystry@WhizFizz":
        st.session_state.authenticated = True
        st.success("✅ Authentication Successful!")
        st.rerun()
    elif passcode:
        st.error("❌ Incorrect Passcode! Try Again.")

# Preprocessing function
def preprocess(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    return text

# Compute similarity function
def compute_similarity(user_input):
    preprocessed_mystery = preprocess(MYSTERY_SENTENCE)
    preprocessed_user = preprocess(user_input)

    mystery_embedding = model.encode(preprocessed_mystery, convert_to_tensor=True, dtype=torch.float64)
    user_embedding = model.encode(preprocessed_user, convert_to_tensor=True, dtype=torch.float64)

    mystery_embedding = normalize(mystery_embedding, p=2, dim=0)
    user_embedding = normalize(user_embedding, p=2, dim=0)

    similarity_score = (mystery_embedding @ user_embedding.T).item()
    
    return similarity_score

# Step 3: Sentence Matching Challenge
if "authenticated" in st.session_state:
    st.subheader("🔍 AI Challenge")
    st.write("Can you guess a sentence similar to the hidden message?")
    user_input = st.text_input("Enter your prompt:")

    if user_input:
        similarity_score = compute_similarity(user_input)
        st.markdown(f"### Similarity Score: `{similarity_score:.4f}`")

        if similarity_score >= 0.95:
            st.success("🎯Perfect match! You've cracked the hidden sentence!")
        elif similarity_score >= 0.85:
            st.success("✅Great match! You are close!")
        elif similarity_score >= 0.75:
            st.warning("🧐Good attempt! Try refining your sentence.")
        elif similarity_score >= 0.75:
            st.warning("Keep on trying.. You can!")
        else:
            st.error("❌Low similarity. Try again!")

        # Store result in leaderboard
        st.session_state.leaderboard.append({
            "Username": st.session_state.username,
            "Score": round(similarity_score, 4)
        })

# Step 4: Leaderboard (Display in Submission Order)
if "leaderboard" in st.session_state and len(st.session_state.leaderboard) > 0:
    st.subheader("🏆 Leaderboard")

    leaderboard_df = pd.DataFrame(st.session_state.leaderboard)
    st.dataframe(leaderboard_df, use_container_width=True)

