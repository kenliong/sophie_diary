import streamlit as st
import os
from utils.llm_utils import *
from utils.prompt_templates import *

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
from old_diary_entries import old_diary_entries
from new_diary_entry import *
import uuid


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


def format_diary_entry(diary_entry):
    """
    Format diary entry for vector store
    """
    #diary_entry_path = "test_diary.txt"
    #diary_entry = TextLoader(diary_entry_path).load()
    diary_with_metadata = {
        "metadata": [{"date": diary_entry["entry_date"]}, {"title": diary_entry["entry_title"]}],
        "diary_content": diary_entry["entry_content"],
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
    sys_prompt = get_chatbot_system_prompt()
    
    model = get_llm_instance()
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
    
    chat_history = get_user_inputs_from_chat_model(chat_model, user_msg)
    conversation_labels = DeepDiveConversationLabels()

    if len(chat_history) > 0:

        # currently we are running this sequentially. Potentially to run this in the "background"
        conversation_labels = extract_info_from_conversation(chat_history)
        if check_conversation_labels(conversation_labels):
            
            # add to vectorstore
            diary_entry_summary = summarize_new_entry(chat_model)
            output_dict = prepare_output_dict(conversation_labels, diary_entry_summary)
            #add_new_diary_to_db(output_dict)
            return 'All values collected.', output_dict
        
    response = chat_model.send_message(user_msg)

    return response.text, conversation_labels.model_dump()

def get_user_inputs_from_chat_model(chat_model, user_msg=''):
    chat_history = ''
    for msg in chat_model.history[2:]:
        if msg.role == 'user':
            chat_history += msg.parts[0].text + '\n\n'

    chat_history += f'{user_msg} \n\n'

    return chat_history

def prepare_output_dict(conversation_labels, diary_entry_summary):
    output_dict = conversation_labels.dict()

    output_dict['entry'] = uuid.uuid4()
    output_dict['entry_date'] = datetime.now().strftime("%Y-%m-%d")
    output_dict['entry_title'] = diary_entry_summary.entry_title
    output_dict['entry_content'] = diary_entry_summary.entry_summary

    return output_dict
