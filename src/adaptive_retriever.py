from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

CHROMA_DIR = "./chroma_db"


class AdaptiveRetriever:
    def __init__(self, documents_path='data/documents'):
        embeddings = OpenAIEmbeddings()

        if Path(CHROMA_DIR).exists():
            print("Loading existing vector database...")
            self.vectordb = Chroma(
                persist_directory=CHROMA_DIR,
                embedding_function=embeddings,
            )
            print("Vector database loaded")
        else:
            print("Loading documents...")
            loader = DirectoryLoader(
                documents_path,
                glob="**/*.txt",
                loader_cls=TextLoader
            )
            documents = loader.load()
            print(f"Loaded {len(documents)} documents")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            self.chunks = text_splitter.split_documents(documents)
            print(f"Created {len(self.chunks)} chunks")

            print("Creating vector database...")
            self.vectordb = Chroma.from_documents(
                documents=self.chunks,
                embedding=embeddings,
                persist_directory=CHROMA_DIR,
            )
            print("Vector database created")

    def retrieve(self, query: str, complexity: str):
        k_map = {
            'simple': 3,
            'medium': 5,
            'complex': 10
        }
        k = k_map.get(complexity, 5)
        print(f"Retrieving {k} chunks for {complexity} query...")
        return self.vectordb.similarity_search(query, k=k)
