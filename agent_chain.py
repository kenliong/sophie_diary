import os
from typing import Dict

import google.generativeai as genai
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from old_diary_entries import old_diary_entries

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def add_old_diary_entries_to_db(old_diary_entries: Dict):
    """
    Add old diary entries to the vector store
    """
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        list_of_documents = []

        for _, diary_entry in old_diary_entries.iterrows():
            list_of_documents.append(
                Document(
                    page_content=diary_entry["entry_desc"],
                    metadata=dict(
                        entry=diary_entry["entry"],
                        date=diary_entry["entry_date"],
                        title=diary_entry["entry_title"],
                    ),
                )
            )
        vector_store = FAISS.from_documents(list_of_documents, embeddings)
        vector_store.save_local("faiss_index")
        return {
            "status": "success",
            "message": "Old diary entries added to the vector store successfully.",
        }
    except Exception as e:
        return {"status": "failure", "message": f"An error occurred: {str(e)}"}


def format_diary_entry(diary_entry_path: str, **kwargs):
    """
    Format diary entry for vector store
    """
    diary_entry_path = "test_diary.txt"
    diary_entry = TextLoader(diary_entry_path).load()
    diary_with_metadata = {
        "metadata": [{"date": diary_entry["datetime"]}, {"title": diary_entry["title"]}],
        "diary_content": diary_entry["content"],
    }
    return diary_with_metadata


def add_new_diary_to_db(diary_entry: Dict):
    """
    'dairy_entry': should be a path to a .txt file
    Add new diary entries to the vector store
    If vector store already exists, add new entries to it
    Otherwise, create a new vector store
    """

    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_store = FAISS.load_local(
        "faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True
    )

    formatted_diary_entry = format_diary_entry(diary_entry)
    vector_store.add_documents(
        documents=formatted_diary_entry["diary_content"],
        metadatas=formatted_diary_entry["metadata"],
    )

    results = vector_store.asimilarity_search(formatted_diary_entry["diary_content"], top_k=1)
    for result in results:
        print(f"Diary content: {result['text']}")
        print(f"Metadata: {result['metadata']}")

    vector_store.save_local("faiss_index")
    return vector_store


def generate_initial_prompts():
    """
    Generates starter prompts based on past diary entries
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_store = FAISS.load_local(
        "faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True
    )
    past_entries = vector_store.search(" ", search_type="similarity", k=5)
    context = "\n".join(entry.page_content for entry in past_entries)

    prompt_template = f"""
    You are a pen pal for a friend who has been struggling with their mental health.
    Based on the following past diary entries, generate 5 relevant questions to prompt your friend to start journalling.
    Past Entries: {context}

    Questions:
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
    chain = LLMChain(llm=model, prompt=prompt)

    result = chain.run({"context": context})
    return result


def chat_chain():
    """
    Uses langchain to create a question-answering chain for chatbot
    """

    # TODO: Implement this function

    return


def summary_prompts():
    """
    Provies a summary of the insights
    1) common topics
    2) actionable insights

    :return:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_store = FAISS.load_local(
        "faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True
    )
    # context is part of the vector store
    past_entries = vector_store.search(" ", search_type="similarity", k=20)
    context = "\n".join(entry.page_content for entry in past_entries)

    prompt_template = "Based on all my journal entries, can I identify 5 common topics?"
    prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
    chain = LLMChain(llm=model, prompt=prompt)
    result_topic = chain.run({"context": context})

    prompt_template = "Based on all my journal entries, can I extract out actionable insights based on frustration?"
    prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
    chain = LLMChain(llm=model, prompt=prompt)
    result_insights = chain.run({"context": context})

    return result_topic, result_insights


def get_llm_chat_instance():
    sys_prompt = """
You are a therapist psychologist, and you have ability like Chris Voss, the FBI hostage negotiator, who labels what this person is going through.

Based on the ongoing conversations, prompt the user and ask questions to get the following information:
1. The emotions that this person experienced
2. The current state (or real outcome) that this person experienced
3. The desired state (or desired outcome, expectation) that this person expected.
4. identify a deep-dive question anchored to this situation. Our goal is to understand what's important to them and why.
5. put all this together. Then make it conversational like Chris Voss.
(i.e. seems like you were disappointed when xyz. looks like you experienced [current state], while you expected [desired state]. [deep-dive question])
    """.strip()

    model = genai.GenerativeModel("gemini-1.5-flash")
    chat = model.start_chat(
        history=[
            {"role": "user", "parts": [sys_prompt]},
            {"role": "model", "parts": ["Understood."]},
        ]
    )

    return chat


def chat_with_user(user_msg, chat_model):
    """
    Takes a user message, creates a response. Will add logic steps to steer the conversation where needed.
    """
    # To do: to explore streaming
    # - https://ai.google.dev/gemini-api/docs/get-started/tutorial?lang=python
    # - https://docs.streamlit.io/develop/api-reference/write-magic/st.write_stream

    response = chat_model.send_message(user_msg)

    return response.text
