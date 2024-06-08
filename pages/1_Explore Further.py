"""
"full table" synthesis, after we collect about 10 daily entries, use LLM to summarise and extract them

"""
import os
import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv

from agent_chain import generate_initial_prompts
from utils import get_custom_css_modifier
from agent_chain import summary_prompts

result_topic, result_insights = summary_prompts()

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

st.set_page_config(
    page_title="Template Chatbot", page_icon="💬", layout="wide", initial_sidebar_state="expanded"
)
st.markdown(get_custom_css_modifier(), unsafe_allow_html=True)

st.markdown("<h5 style='text-align: left;'>💬 Template Chatbot</h5>", unsafe_allow_html=True)
st.title("Sophie's Diary - Explore Further")

st.header("Full Table Synthesis We have taken a deeper dive at your entries and here's a summary!")

st.subheader("Common patterns")
st.write(result_topic)

st.subheader("Actionable Insights")
st.write(result_insights)