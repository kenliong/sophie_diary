import hmac
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

st.session_state["authenticated"] = False

def check_password():
    """Returns `True` if the user had a correct password."""

    def login_form():
        """Form with widgets to collect user information"""
        with st.form("Credentials"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.form_submit_button("Log in", on_click=password_entered)
        st.button("Sign Up", on_click=sign_up)
    
    def sign_up():
        """Allows new users to sign up"""
        with st.form("Signup"):
            st.text_input("Username", key="su_username")
            st.text_input("Password", type="password", key="su_password")
            st.text_input("Reenter Password", type="password", key="rep_password")
            st.form_submit_button("Set up account", on_click=set_up_account)
        
    def set_up_account():
        if st.session_state["su_username"] in st.secrets[
            "passwords"
        ]:
            st.error("😕 User already exists")
        elif not hmac.compare_digest(
            st.session_state["su_password"],
            st.session_state["rep_password"],
        ):
            st.error("😕 Passwords do not match")
        else:
            st.warning('You have signed up!')

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["username"] in st.secrets[
            "passwords"
        ] and hmac.compare_digest(
            st.session_state["password"],
            st.secrets.passwords[st.session_state["username"]],
        ):
            st.session_state["password_correct"] = True
            st.session_state["authenticated_user"] = st.session_state["username"]
            st.session_state["user_journal_path"] = f'data/journal_entries/{st.session_state["username"]}.csv'
            del st.session_state["password"]  # Don't store the username or password.
            del st.session_state["username"]
            
        else:
            st.session_state["password_correct"] = False

    # Return True if the username + password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show inputs for username + password.
    login_form()
    if "password_correct" in st.session_state: #If the key exists: This means the user has previously attempted to log in. 
        st.error("😕 User not known or password incorrect")
    return False


if not check_password():
    st.stop()
st.session_state["authenticated"] = True
st.markdown(get_custom_css_modifier(), unsafe_allow_html=True)

st.markdown("<h4 style='text-align: left;'>💬 Sophie's Diary</h4>", unsafe_allow_html=True)

with st.form(key="new_entry_form", clear_on_submit=False):
    new_entry_text = st.text_area("What's on your mind?", "", key="new_entry_text")

    new_entry_submit = st.form_submit_button(label="Submit")

if new_entry_submit and len(new_entry_text.strip()) == 0:
    st.error("Please key a new entry", icon="⚠️")
elif new_entry_submit:
    #st.button("Explore further", key="explore_further", on_click=enable_explore_further)
    enable_explore_further()

with st.expander("Debug view"):
    if st.session_state["conversation_labels"]:
        st.write(st.session_state["conversation_labels"])

    if st.session_state["chat_model"]:
        st.write(st.session_state["chat_model"].history[0].parts[0].text)

if st.session_state["explore_further_enabled"]:
    chat_history_box = st.container()

    with chat_history_box:
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

    prompt = st.chat_input()
    if prompt:
        st.chat_message("user").write(prompt)

        response, conversation_labels = chat_with_user(prompt)

        st.session_state["conversation_labels"] = conversation_labels

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": response})

        st.chat_message("assistant").write(response)
        st.rerun()
