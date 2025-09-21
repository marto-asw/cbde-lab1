from sentence_transformers import SentenceTransformer
from connect import connect
import time
import numpy as np


#connection to the db
conn = connect()
cur = conn.cursor()

# Create embeddings table if it doesn't exist
cur.execute("""
CREATE TABLE IF NOT EXISTS embeddings (
    sentence_id INTEGER PRIMARY KEY,
    embedding DOUBLE PRECISION[],
    CONSTRAINT embeddings_sentence_id_fkey FOREIGN KEY (sentence_id)
        REFERENCES sentences(id)
        ON DELETE CASCADE
);
""")
conn.commit()

# Fetch sentences from the database
cur.execute("SELECT id, text FROM sentences ORDER BY id;")
rows = cur.fetchall()
sentence_ids = [row[0] for row in rows]  # IDs to link embeddings
sentences = [row[1] for row in rows]     # Actual text

# Load model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Create embeddings
embeddings = model.encode(sentences, batch_size=32, show_progress_bar=True)

#print("Shape de los embeddings:", embeddings.shape)  # (10000, 384)

#insrert embeddings
embedding_times = []
for sentence_id, emb in zip(sentence_ids, embeddings):
    start = time.time()
    emb_list = emb.tolist()
    cur.execute("INSERT INTO embeddings (sentence_id, embedding) VALUES (%s, %s);", 
                (sentence_id, emb_list))
    conn.commit()
    embedding_times.append(time.time() - start)

print("Tiempos inserción embeddings:", 
      f"min={np.min(embedding_times)} max={np.max(embedding_times)} avg={np.mean(embedding_times)} std={np.std(embedding_times)}")
