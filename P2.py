from sentence_transformers import SentenceTransformer
from connect import connect
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

conn = connect()
cur = conn.cursor()

# Recuperar todas las oraciones y embeddings
cur.execute("SELECT s.id, s.text, e.embedding FROM sentences s JOIN embeddings e ON s.id = e.sentence_id;")
rows = cur.fetchall()

ids = [r[0] for r in rows]
texts = [r[1] for r in rows]
embs = np.array([r[2] for r in rows])

# Choose 10 random sentences to test
test_indices = np.random.choice(len(texts), size=10, replace=False)
print(f"The indices are:",test_indices)
print("=== 10 frases seleccionadas aleatoriamente ===")
for i, idx in enumerate(test_indices, 1):
    print(f"{i}. {texts[idx]}")

#insert senteces
times = []

for idx in test_indices:
    start = time.time() 
    query_emb = embs[idx].reshape(1, -1)

    # Cosine similarity
    cos_sim = cosine_similarity(query_emb, embs)[0]
    top2_cos = np.argsort(cos_sim)[-3:-1][::-1]  

    # Euclidean distance
    euc_dist = euclidean_distances(query_emb, embs)[0]
    top2_euc = np.argsort(euc_dist)[1:3]

    times.append(time.time() - start)

    print(f"\nSentence: {texts[idx]}")
    print("Top-2 (cosine):", [texts[i] for i in top2_cos])
    print("Top-2 (euclidean):", [texts[i] for i in top2_euc])

print("Computation times:", 
      f"min={np.min(times)} max={np.max(times)} avg={np.mean(times)} std={np.std(times)}")
