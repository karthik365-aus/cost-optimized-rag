from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Load existing database
vectordb = Chroma(
    persist_directory="./chroma_db",
    embedding_function=OpenAIEmbeddings()
)

# Test queries
results = vectordb.similarity_search("office hours", k=3)
for doc in results:
    print(doc.page_content)