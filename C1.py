
from sentence_transformers import SentenceTransformer
import time
import numpy as np
import chromadb
from chromadb.config import Settings

chroma_client = chromadb.PersistentClient(
    path="./chroma_data",  # where the database is stored   
    settings=Settings(),
)

collection = chroma_client.get_collection("book_corpus")

# --- Load sentences from Chroma ---
# Fetch all documents (sentences) and their IDs
results = collection.get(include=["documents", "metadatas"])
sentences = results["documents"]
ids = [m["id"] for m in results["metadatas"]]

print(f"Loaded {len(sentences)} sentences from Chroma.")

# Load the model 
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Create embeddings
batch_size = 5000  # keep same batch size as before
embeddings = []
times_embs = []

for i in range(0, len(sentences), batch_size):
    start = time.time()
    
    batch_sentences = sentences[i:i+batch_size]
    batch_ids = ids[i:i+batch_size]
    
    batch_embeddings = model.encode(batch_sentences, batch_size=32, show_progress_bar=True).tolist()
    embeddings.extend(batch_embeddings)
    
    # Store embeddings in Chroma
    collection.add(
        embeddings=batch_embeddings,
        ids=batch_ids
    )
    
    end = time.time()
    times_embs.append(end - start)

print("Embedding storage times:")
print(f"  min = {np.min(times_embs):.4f}")
print(f"  max = {np.max(times_embs):.4f}")
print(f"  avg = {np.mean(times_embs):.4f}")
print(f"  std = {np.std(times_embs):.4f}")

print(f"Number of sentences in Chroma now: {collection.count()}")
