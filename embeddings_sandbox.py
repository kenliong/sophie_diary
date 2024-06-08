from langchain_community.document_loaders import TextLoader
# from langchain_openai import OpenAIEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import utils.journal_query as jq
import utils.prompt_templates as pt
import os
import google.generativeai as genai
from dotenv import load_dotenv
from langchain.chains import LLMChain


load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# from old_diary_entries import old_diary_entries
# from agent_chain import add_old_diary_entries_to_db
#setup db
# status = add_old_diary_entries_to_db(old_diary_entries)
# print(status)
model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3) #this should be abstracted later
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004") #this should be abstracted later
db = jq.get_db(embeddings)

#construct journal query
query = pt.get_journal_query_topic_based().format(topics='family')

docs, sims  = jq.get_docs_with_query(db, query, num_of_docs = 4)

print(len(docs))
print(len(sims))
print(sims)
# print(docs[0])
# print(docs[1])

prompt = pt.prompt_on_docs()
chain = LLMChain(llm=model, prompt=prompt)
result = chain.run({"docs": docs})
print(result)

print("done")