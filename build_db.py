import sqlite3
import os
import nltk
from nltk.corpus import wordnet as wn

print("Downloading WordNet data...")
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

db_path = "revlex.db"
if os.path.exists(db_path):
    os.remove(db_path)

print(f"Creating SQLite database: {db_path}...")
conn = sqlite3.connect(db_path)
cur = conn.cursor()

# 1. Main structured table for fast row lookups
cur.execute("""
CREATE TABLE IF NOT EXISTS entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    word TEXT NOT NULL,
    pos TEXT NOT NULL,
    definition TEXT NOT NULL,
    synonyms TEXT
);
""")

# 2. Virtual FTS5 table for fast reverse text search with BM25 ranking
cur.execute("""
CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts USING fts5(
    word,
    pos UNINDEXED,
    definition,
    synonyms,
    content='entries',
    content_rowid='id'
);
""")

print("Extracting synsets and indexing entries...")

pos_map = {
    'n': 'noun',
    'v': 'verb',
    'a': 'adj',
    's': 'adj',
    'r': 'adv'
}

batch_data = []
count = 0

for synset in wn.all_synsets():
    pos_str = pos_map.get(synset.pos(), 'other')
    definition = synset.definition()
    
    # Get all lemmas/synonyms in this synset
    lemmas = [lemma.name().replace('_', ' ') for lemma in synset.lemmas()]
    synonyms_str = ", ".join(lemmas)

    # Add each distinct lemma as a target word mapped to this definition
    for lemma in lemmas:
        if len(lemma) > 1:
            batch_data.append((lemma.lower(), pos_str, definition, synonyms_str))
            count += 1

    if len(batch_data) >= 20000:
        cur.executemany("INSERT INTO entries (word, pos, definition, synonyms) VALUES (?, ?, ?, ?)", batch_data)
        conn.commit()
        batch_data = []
        print(f"Indexed {count:,} definition-word pairs...")

if batch_data:
    cur.executemany("INSERT INTO entries (word, pos, definition, synonyms) VALUES (?, ?, ?, ?)", batch_data)
    conn.commit()

print("Synchronizing Full-Text Search (FTS5) index...")
cur.execute("""
INSERT INTO entries_fts(rowid, word, pos, definition, synonyms)
SELECT id, word, pos, definition, synonyms FROM entries;
""")
conn.commit()

# Optimize FTS index to minimize file size on disk
cur.execute("INSERT INTO entries_fts(entries_fts) VALUES('optimize');")
conn.commit()

# Measure database footprint
file_size_mb = os.path.getsize(db_path) / (1024 * 1024)
print(f"\nDone! Indexed {count:,} definition mappings.")
print(f"Final SQLite database size: {file_size_mb:.2f} MB")

# Quick test query directly via SQLite
print("\nTesting reverse lookup for: 'smell after rain'...")
cur.execute("""
SELECT word, pos, definition, bm25(entries_fts) AS rank 
FROM entries_fts 
WHERE entries_fts MATCH 'smell AND rain' 
ORDER BY rank 
LIMIT 3;
""")
results = cur.fetchall()
for row in results:
    print(f" -> {row[0]} ({row[1]}): {row[2]}")

conn.close()
