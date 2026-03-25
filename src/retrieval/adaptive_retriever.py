from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import os

class AdaptiveRetriever:
    def __init__(self, documents_path='data/documents'):
        """Initialize the retriever with documents"""
        print("Loading documents...")
        
        # Load all .txt files
        loader = DirectoryLoader(
            documents_path,
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        documents = loader.load()
        print(f"✅ Loaded {len(documents)} documents")
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.chunks = text_splitter.split_documents(documents)
        print(f"✅ Created {len(self.chunks)} chunks")
        
        # Create vector database (needs OpenAI API key or use local embeddings)
        # For now, we'll set this up in next step
        
        # Create vector database
        from dotenv import load_dotenv
        load_dotenv()  # Load API key from .env

        print("Creating vector database...")
        embeddings = OpenAIEmbeddings()
        self.vectordb = Chroma.from_documents(
            documents=self.chunks,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        print("✅ Vector database created")
    
    def retrieve(self, query, complexity):
        """
        Retrieve documents based on query and complexity
        
        Args:
            query: The search query
            complexity: 'simple', 'medium', or 'complex'
        
        Returns:
            List of relevant documents
        """
        # Set k based on complexity
        k_map = {
            'simple': 3,
            'medium': 5,
            'complex': 10
        }
        k = k_map.get(complexity, 5)
        
        print(f"Retrieving {k} documents for {complexity} query...")
        
        # TODO: Implement actual vector search in next step
        # For now, return mock result
        
        # Actual vector search
        if self.vectordb:
            docs = self.vectordb.similarity_search(query, k=k)
            return docs
        else:
            return self.chunks[:k]


# Test the code
if __name__ == "__main__":
    # Create retriever
    retriever = AdaptiveRetriever()
    
    # Test with different complexity levels
    test_query = "What are the office hours?"
    
    for complexity in ['simple', 'medium', 'complex']:
        docs = retriever.retrieve(test_query, complexity)
        print(f"\n{complexity.upper()}: Retrieved {len(docs)} documents")