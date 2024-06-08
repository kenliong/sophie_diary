import os
from typing import Dict

import google.generativeai as genai
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from old_diary_entries import emotions, key_topics, mental_tendencies, old_diary_entries
from utils.prompt_templates import (
    generate_emotions_template,
    generate_key_topics_template,
    generate_mental_tendencies_template,
)

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def generate_mental_tendencies(diary_entry):
    """
    Uses gemini to tag mental tendencies to diary entries
    """
    entry_content = diary_entry["entry_content"]
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt_template = generate_mental_tendencies_template()
    prompt_template.format(entry_content=entry_content, mental_tendencies=mental_tendencies)
    chain = LLMChain(llm=model, prompt=prompt_template)
    inputs = {"entry_content": entry_content, "mental_tendencies": mental_tendencies}
    response = chain.run(inputs)

    return response


def generate_emotions(diary_entry):
    """
    Uses gemini to tag emotions to diary entries
    """
    entry_content = diary_entry["entry_content"]
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt_template = generate_emotions_template()
    prompt_template.format(entry_content=entry_content, emotions=emotions)
    chain = LLMChain(llm=model, prompt=prompt_template)
    inputs = {"entry_content": entry_content, "emotions": emotions}
    response = chain.run(inputs)

    return response


def generate_key_topics(diary_entry):
    """
    Uses gemini to generate key topics to diary entries
    """
    entry_content = diary_entry["entry_content"]
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt_template = generate_key_topics_template()
    prompt_template.format(entry_content=entry_content, key_topics=key_topics)
    chain = LLMChain(llm=model, prompt=prompt_template)
    inputs = {"entry_content": entry_content, "key_topics": key_topics}
    response = chain.run(inputs)

    return response


def generate_analytics_old_entries(diary_entry: Dict):
    emotions = generate_emotions(diary_entry)
    key_topics = generate_key_topics(diary_entry)
    mental_tendencies = generate_mental_tendencies(diary_entry)

    return emotions, key_topics, mental_tendencies


def add_new_diary_to_db_and_csv(diary_entry: Dict):
    """
    Add new diary entries to the vector store and csv file
    """
    csv_data = pd.read_csv("data/journal_entries_v2.csv")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_store = FAISS.load_local(
        "faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True
    )

    document = Document(page_content=diary_entry.pop("entry_content"), metadata=diary_entry)
    vector_store.add_documents([document])
    vector_store.save_local("faiss_index")
    print("added to vectorstore")

    new_row_in_csv = len(csv_data)
    csv_data.loc[new_row_in_csv] = diary_entry
    csv_data.to_csv("data/journal_entries_v2.csv", index=False)
    print("added to csv")


def generate_analytics_new_entry(output_dict: Dict):
    emotions = generate_emotions(output_dict)
    key_topics = generate_key_topics(output_dict)
    mental_tendencies = generate_mental_tendencies(output_dict)
    output_dict["emotions"] = emotions
    output_dict["key_topics"] = key_topics
    output_dict["mental_tendencies"] = mental_tendencies

    add_new_diary_to_db_and_csv(output_dict)
