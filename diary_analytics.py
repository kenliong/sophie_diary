import os

import google.generativeai as genai
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI

from old_diary_entries import emotions, old_diary_entries
from utils.prompt_templates import generate_emotions_from_diary_template

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def generate_emotions():
    """
    Uses gemini to tag emotions to diary entries
    """
    for _, diary_entry in old_diary_entries.iterrows():
        entry_content = diary_entry["entry_content"]
        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        prompt_template = generate_emotions_from_diary_template()
        prompt_template.format(entry_content=entry_content, emotions=emotions)
        chain = LLMChain(llm=model, prompt=prompt_template)
        inputs = {"entry_content": entry_content, "emotions": emotions}
        response = chain.run(inputs)
        print(response)
