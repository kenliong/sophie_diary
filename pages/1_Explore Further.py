"""
"full table" synthesis, after we collect about 10 daily entries, use LLM to summarise and extract them

"""
import os
import google.generativeai as genai
import streamlit as st
import utils.journal_query as jq
from dotenv import load_dotenv

#from agent_chain import generate_initial_prompts
#from agent_chain import summary_prompts
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from utils.llm_utils import get_sahha_insights

#result_topic, result_insights = summary_prompts()

if st.session_state["authenticated"]  == False:
    st.title(f'Please login in the "main" tab first')
    st.stop()

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

st.set_page_config(
    page_title="Template Chatbot", page_icon="💬", layout="wide", initial_sidebar_state="expanded"
)
#st.markdown(get_custom_css_modifier(), unsafe_allow_html=True)

st.markdown("<h5 style='text-align: left;'>💬 Template Chatbot</h5>", unsafe_allow_html=True)
st.title("Sophie's Diary - Explore Further")

st.header("Full Table Synthesis We have taken a deeper dive at your entries and here's a summary!")

st.subheader("Actionable Insights")

model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
# vector_store = FAISS.load_local(
#     f'faiss_index/{st.session_state["authenticated_user"]}', embeddings=embeddings, allow_dangerous_deserialization=True
# )
vector_store = jq.get_db(st.session_state["authenticated_user"], embeddings)

# context is part of the vector store
past_entries = vector_store.search(" ", search_type="similarity", k=4)
context = "\n".join(entry.page_content for entry in past_entries)


sahha_prompt, well_being_score = get_sahha_insights(1,1)

prompt_template = f"""
I want to extract 5 actionable insights from the following data. Each point should be less than 20 words.
These are some of my past journal entries: {context}.
And this is some data of my daily activities: {sahha_prompt}
"""
image_placeholder = st.empty()

with st.spinner('Loading from RAG...'):
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "sahha_prompt"])
    prompt.format(context=context, sahha_prompt=sahha_prompt)
    chain = LLMChain(llm=model, prompt=prompt)
    inputs = {"context": context, "sahha_prompt": sahha_prompt}
    result_insights = chain.run(inputs)

st.write(result_insights)