import os

import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv

from agent_chain import generate_initial_prompts
from utils import get_custom_css_modifier

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

st.set_page_config(
    page_title="Template Chatbot", page_icon="💬", layout="wide", initial_sidebar_state="expanded"
)
st.markdown(get_custom_css_modifier(), unsafe_allow_html=True)

st.markdown("<h5 style='text-align: left;'>💬 Template Chatbot</h5>", unsafe_allow_html=True)
st.title("Sophie's Diary")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
if "starting_prompts" not in st.session_state:
    st.session_state["starting_prompts"] = generate_initial_prompts()
    st.session_state["starting_prompts_shown"] = False

# Display starting prompts only once before the chat starts
if not st.session_state["starting_prompts_shown"]:
    st.session_state["starting_prompts_shown"] = True
    st.markdown(
        "<div class='suggested-prompts'><h4>Suggested Prompts</h4></div>", unsafe_allow_html=True
    )
    for prompt in st.session_state["starting_prompts"].split("\n"):
        if prompt.strip():
            st.write(f"- {prompt}")

chat_history_box = st.container()
with chat_history_box:
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])


prompt = st.chat_input()
if prompt:
    st.chat_message("user").write(prompt)

    response = f"I hear you said {prompt}"

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": response})

    st.chat_message("assistant").write(response)
    st.rerun()
