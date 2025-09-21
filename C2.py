import chromadb
from chromadb.config import Settings
import time
import numpy as np

chroma_client = chromadb.PersistentClient(
    path="./chroma_data", 
    settings=Settings(),
)
collection = chroma_client.get_collection("book_corpus")

# Indices of the 10 sentences used in Postgres
query_indices = [2025, 9549, 8051, 2114, 3207, 8506, 2689, 554, 7355, 9761]
query_ids = [f"sentence_{i}" for i in query_indices] 
query_data = collection.get(ids=query_ids, include=["documents", "embeddings"])

query_sentences = query_data["documents"]
query_embeddings = query_data["embeddings"]

print("=== Query Sentences ===")
for i, s in zip(query_indices, query_sentences):
    print(f"[{i}] {s[:80]}...")  # truncate for readability

times_cosine = []
times_euclidean = []

# --- Query with COSINE similarity ---
print("\n=== Top-2 Results (Cosine) ===")
for i, (q_text, q_emb) in enumerate(zip(query_sentences, query_embeddings)):
    start = time.time()  
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=3  
    )
    elapsed = time.time() - start
    times_cosine.append(elapsed)

    returned = results["documents"][0]
    returned_ids = results["ids"][0]

    # Filter out the query itself
    filtered = [(id_, doc) for id_, doc in zip(returned_ids, returned) if doc != q_text][:2]

    print(f"\nQuery [{query_indices[i]}]: {q_text}")
    for rank, (id_, doc) in enumerate(filtered, start=1):
        print(f"  {rank}. ({id_}) {doc}")

# --- Query with EUCLIDEAN distance ---
print("\n=== Top-2 Results (Euclidean) ===")
for i, (q_text, q_emb) in enumerate(zip(query_sentences, query_embeddings)):
    start = time.time()  
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=3
    )
    elapsed = time.time() - start
    times_euclidean.append(elapsed)

    returned = results["documents"][0]
    returned_ids = results["ids"][0]

    filtered = [(id_, doc) for id_, doc in zip(returned_ids, returned) if doc != q_text][:2]

    print(f"\nQuery [{query_indices[i]}]: {q_text}")
    for rank, (id_, doc) in enumerate(filtered, start=1):
        print(f"  {rank}. ({id_}) {doc}")

# Estadísticas de tiempo
print("\nComputation times (Cosine):",
      f"min={np.min(times_cosine):.6f}s max={np.max(times_cosine):.6f}s avg={np.mean(times_cosine):.6f}s std={np.std(times_cosine):.6f}s")
print("Computation times (Euclidean):",
      f"min={np.min(times_euclidean):.6f}s max={np.max(times_euclidean):.6f}s avg={np.mean(times_euclidean):.6f}s std={np.std(times_euclidean):.6f}s")