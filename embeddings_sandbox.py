from langchain_community.document_loaders import TextLoader
# from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from agent_chain import add_diary_to_vector_store

diary1 = TextLoader('test_diary.txt').load()
diary2 = TextLoader('test_diary.txt').load()
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=len(raw), chunk_overlap=0)
 
#day 0
dairy_list  = []

#day 1
dairy_list =  [diary1[0]]

#day 2
dairy_list.append(diary2[0])

# print(type(raw_documents))
# print(len(raw_documents))
# print(type(raw_documents[0]))

print("-"*6)
print(len(dairy_list))
print(type(dairy_list[0]))
print(type(dairy_list[1]))

print("-"*6)
print(diary1[0].metadata)

# add_diary_to_vector_store('test_diary_3.txt')

model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
db = FAISS.load_local(
    "faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True
)
db.add_documents(TextLoader("test_diary_2.txt").load())
db.add_documents(TextLoader("test_diary_3.txt").load())
# docs = db.similarity_search_with_relevance_scores(query, K=4)
# from utils.journal_query import get_docs_with_query
import utils.journal_query as jq


#construct journal query

query = jq.get_journal_query_simple()
docs = jq.get_docs_with_query(db, query, num_of_docs = 4, score_threshold = 0.1)
print(len(docs))
print(docs[0])
print(docs[1])
print("done")
