
from datasets import load_dataset
import time
import numpy as np
import chromadb
from chromadb.config import Settings

chroma_client = chromadb.PersistentClient(
    path="./chroma_data",  # where the database will be stored
    settings=Settings(),
)

collection = chroma_client.create_collection(name="book_corpus")

#Download BookCorpus
dataset = load_dataset("bookcorpus", split="train", trust_remote_code=True)


#Select the first 10,000 sentences with the same seed for reproducibility
sample = dataset.shuffle(seed=42).select(range(10000))
sentences = [x["text"] for x in sample]
ids = [f"sentence_{i}" for i in range(len(sentences))]

# --- Measure storing documents ---
times_docs = []
batch_size = 5000

for i in range(0, len(sentences), batch_size):
    start = time.time()
    batch_sentences = sentences[i:i+batch_size]
    batch_ids = ids[i:i+batch_size]
    
    collection.add(
        documents=batch_sentences,
        ids=batch_ids,
        metadatas=[{"id": id_} for id_ in batch_ids]  # store IDs in metadata
    )
    
    end = time.time()
    times_docs.append(end - start)

print("Text insertion times:")
print(f"  min = {np.min(times_docs):.4f}")
print(f"  max = {np.max(times_docs):.4f}")
print(f"  avg = {np.mean(times_docs):.4f}")
print(f"  std = {np.std(times_docs):.4f}")
