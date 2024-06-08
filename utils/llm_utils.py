import asyncio
import os
from datetime import datetime, timezone
from typing import Dict

import google.generativeai as genai
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.output_parsers import CommaSeparatedListOutputParser, PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader

from utils.prompt_templates import *

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_llm_instance():
    model = genai.GenerativeModel("gemini-1.5-flash")
    # model = genai.GenerativeModel("gemini-1.0-pro")
    return model


def get_completion(model, prompt):
    response = model.generate_content(prompt)
    return response.text


async def async_get_completion(model, prompt):
    response = await model.generate_content_async(prompt)
    return response.text
