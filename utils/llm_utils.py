import os
import google.generativeai as genai
from langchain.output_parsers import CommaSeparatedListOutputParser, PydanticOutputParser
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv

from typing import Dict

from utils.prompt_templates import *
import asyncio
from datetime import datetime, timezone

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_llm_instance():
    model = genai.GenerativeModel('gemini-1.5-flash')
    #model = genai.GenerativeModel('gemini-1.0-pro')
    return model

def get_completion(model,prompt):
    response = model.generate_content(prompt)
    return response.text

async def async_get_completion(model, prompt):
    response = await model.generate_content_async(prompt)
    return response.text