from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

embeddings = OllamaEmbeddings(model=os.getenv("EMBEDDING_MODEL"))

vector_store = Chroma(
    collection_name=os.getenv("COLLECTION_NAME"),
    embedding_function=embeddings,
    persist_directory=os.getenv("DATABASE_LOCATION"),
)

result = vector_store.get(include=["documents", "metadatas", "embeddings"])

print(f"Total chunks stored: {len(result['ids'])}\n")

for i in range(len(result["ids"])):
    print("ID:", result["ids"][i])
    print("Source:", result["metadatas"][i]["source"])
    print("Text preview:", result["documents"][i][:80], "...")
    print("Embedding dims:", len(result["embeddings"][i]))
    print("First 5 numbers:", result["embeddings"][i][:5])
    print("---")