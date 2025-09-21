from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from connect import connect
import time
import numpy as np

# Download BookCorpus
dataset = load_dataset("bookcorpus", split="train", trust_remote_code=True)

# Chunks selected
sample = dataset.shuffle(seed=42).select(range(10000))
sentences = [x["text"] for x in sample]

#connection to the db
conn = connect()
cur = conn.cursor()

# Create table if it does not exist
cur.execute("""
CREATE TABLE IF NOT EXISTS sentences (
    id SERIAL PRIMARY KEY,
    text TEXT NOT NULL
);
""")
conn.commit()

#insert senteces
insert_times = []
for sentence in sentences:
    start = time.time()
    cur.execute("INSERT INTO sentences (text) VALUES (%s) RETURNING id;", (sentence,))
    conn.commit()
    insert_times.append(time.time() - start)

print("Text insertion times:" 
      f"min={np.min(insert_times)} max={np.max(insert_times)} avg={np.mean(insert_times)} std={np.std(insert_times)}")

