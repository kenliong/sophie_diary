from langchain_community.document_loaders import TextLoader
# from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

raw_documents = TextLoader('test_diary.txt').load()
text_splitter = CharacterTextSplitter(chunk_size=1, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
# print(type(raw_documents))
# print(len(raw_documents))
# print(type(raw_documents[0]))

print("-"*6)
print(documents)
print(type(documents))
print(len(documents))
print(type(documents[0]))

