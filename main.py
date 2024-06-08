import streamlit as st
from dotenv import load_dotenv

from utils.llm_utils import *
from utils.streamlit_utils import *

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
load_resources()

st.set_page_config(
    page_title="Sophie's Diary", page_icon="💬", layout="wide", initial_sidebar_state="expanded"
)
st.markdown(get_custom_css_modifier(), unsafe_allow_html=True)

st.markdown("<h4 style='text-align: left;'>💬 Sophie's Diary</h4>", unsafe_allow_html=True)

with st.form(key="new_entry_form", clear_on_submit=False):
    new_entry_text = st.text_area("What's on your mind?", "", key="new_entry_text")

    new_entry_submit = st.form_submit_button(label="Submit")

if new_entry_submit and len(new_entry_text.strip()) == 0:
    st.error("Please key a new entry", icon="⚠️")
elif new_entry_submit:
    st.button("Explore further", key="explore_further", on_click=enable_explore_further)

if st.session_state["conversation_labels"]:
    st.write(st.session_state["conversation_labels"])

if st.session_state["explore_further_enabled"]:
    chat_history_box = st.container()

    with chat_history_box:
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

    prompt = st.chat_input()
    if prompt:
        st.chat_message("user").write(prompt)

        response, conversation_labels = chat_with_user(prompt, st.session_state["chat_model"])

        st.session_state["conversation_labels"] = conversation_labels

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": response})

        st.chat_message("assistant").write(response)
        st.rerun()
