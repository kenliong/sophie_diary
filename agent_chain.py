import os
import uuid
from datetime import datetime

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from diary_analytics import generate_analytics_new_entry
from new_diary_entry import *
from old_diary_entries import old_diary_entries
from utils.llm_utils import *
from utils.prompt_templates import *

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
                    page_content=diary_entry["entry_content"],
                    metadata=dict(
                        entry=diary_entry["entry"],
                        current_state=diary_entry["current_state"],
                        desired_state=diary_entry["desired_state"],
                        date=diary_entry["entry_date"],
                        title=diary_entry["entry_title"],
                        mental_tendencies=diary_entry["mental_tendencies"],
                        emotions=diary_entry["emotions"],
                        key_topics=diary_entry["key_topics"],
                        reflection_questions=diary_entry["reflection_questions"],
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


def get_llm_chat_instance(system_prompt, previous_chat_model=None):
    chat_history = []
    if previous_chat_model:
        chat_history = previous_chat_model.history[2:]

    model = get_llm_instance()
    chat = model.start_chat(
        history=[
            {"role": "user", "parts": [system_prompt]},
            {"role": "model", "parts": ["Understood."]},
        ]
        + chat_history
    )

    return chat


def chat_with_user(user_msg):
    """
    Takes a user message, creates a response. Will add logic steps to steer the conversation where needed.
    """
    # To do: to explore streaming
    # - https://ai.google.dev/gemini-api/docs/get-started/tutorial?lang=python
    # - https://docs.streamlit.io/develop/api-reference/write-magic/st.write_stream

    chat_model = st.session_state["chat_model"]
    chat_history = get_user_inputs_from_chat_model(chat_model, user_msg).strip()

    conversation_labels = DeepDiveConversationLabels()

    if len(chat_history.strip()) > len(user_msg.strip()):
        # currently we are running this sequentially. Potentially to run this in the "background"?
        conversation_labels = extract_info_from_conversation(chat_history)

        if not conversation_labels.emotions or conversation_labels.emotions == ["None"]:
            new_sys_prompt = get_chatbot_system_prompt(sahha_insights=get_sahha_insights(1, 1))

            st.session_state["chat_model"] = get_llm_chat_instance(new_sys_prompt, chat_model)
            chat_model = st.session_state["chat_model"]

        elif not conversation_labels.current_state or conversation_labels.current_state == "None":
            new_sys_prompt = get_chatbot_system_prompt(
                additional_info="- The current state (or real outcome) that this person experienced",
                sahha_insights=get_sahha_insights(1, 1),
                similar_issues=get_db_context(chat_history),
            )

            st.session_state["chat_model"] = get_llm_chat_instance(new_sys_prompt, chat_model)
            chat_model = st.session_state["chat_model"]

        elif not conversation_labels.desired_state or conversation_labels.desired_state == "None":
            new_sys_prompt = get_chatbot_system_prompt(
                additional_info="- The desired state (or desired outcome, expectation) that this person expected.",
                sahha_insights=get_sahha_insights(1, 1),
                similar_issues=get_db_context(chat_history),
            )

            st.session_state["chat_model"] = get_llm_chat_instance(new_sys_prompt, chat_model)
            chat_model = st.session_state["chat_model"]

        else:
            # add to vectorstore
            diary_entry_summary = summarize_new_entry(chat_model)
            output_dict = prepare_output_dict(conversation_labels, diary_entry_summary)
            st.session_state["output_complete_flag"] = "True"
            generate_analytics_new_entry(output_dict)
            return (
                "Thanks for sharing! You've finished your reflection and submitted a new diary entry.",
                output_dict,
            )

    response = chat_model.send_message(user_msg)

    return response.text, conversation_labels.model_dump()


def get_user_inputs_from_chat_model(chat_model, user_msg=""):
    chat_history = ""
    for msg in chat_model.history[2:]:
        if msg.role == "user":
            chat_history += msg.parts[0].text + "\n\n"

    chat_history += f"{user_msg} \n\n"

    return chat_history


def prepare_output_dict(conversation_labels, diary_entry_summary):
    output_dict = conversation_labels.dict()

    output_dict["entry"] = uuid.uuid4()
    output_dict["entry_date"] = datetime.now().strftime("%Y-%m-%d")
    output_dict["entry_title"] = diary_entry_summary.entry_title
    output_dict["entry_content"] = diary_entry_summary.entry_summary

    return output_dict
