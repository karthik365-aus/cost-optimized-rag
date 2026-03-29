from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

vectordb = Chroma(
    persist_directory="./chroma_db",
    embedding_function=OpenAIEmbeddings()
)

test_queries = [
    "What are the office hours?",
    "Who founded Notre Dame?",
    "What is the oldest structure at Notre Dame?",
]

for query in test_queries:
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    results = vectordb.similarity_search(query, k=3)
    for i, doc in enumerate(results, 1):
        print(f"\nChunk {i}: {doc.page_content[:200]}...")
