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

conn = sqlite3.connect(db_path)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    word TEXT NOT NULL,
    pos TEXT NOT NULL,
    definition TEXT NOT NULL,
    synonyms TEXT
);
""")

# Enable the 'porter' tokenizer so words match regardless of plural/tense endings
cur.execute("""
CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts USING fts5(
    word,
    pos UNINDEXED,
    definition,
    synonyms,
    content='entries',
    content_rowid='id',
    tokenize='porter ascii'
);
""")

print("Indexing synsets with definitions and usage examples...")

pos_map = {'n': 'noun', 'v': 'verb', 'a': 'adj', 's': 'adj', 'r': 'adv'}
batch_data = []
count = 0

for synset in wn.all_synsets():
    pos_str = pos_map.get(synset.pos(), 'other')
    definition = synset.definition()
    
    # Append usage examples to the searchable text block
    examples = synset.examples()
    if examples:
        definition += " | Examples: " + "; ".join(examples)
        
    lemmas = [l.name().replace('_', ' ') for l in synset.lemmas()]
    synonyms_str = ", ".join(lemmas)

    for lemma in lemmas:
        if len(lemma) > 1:
            batch_data.append((lemma.lower(), pos_str, definition, synonyms_str))
            count += 1

    if len(batch_data) >= 20000:
        cur.executemany("INSERT INTO entries (word, pos, definition, synonyms) VALUES (?, ?, ?, ?)", batch_data)
        conn.commit()
        batch_data = []

if batch_data:
    cur.executemany("INSERT INTO entries (word, pos, definition, synonyms) VALUES (?, ?, ?, ?)", batch_data)
    conn.commit()

print("Synchronizing FTS5 index...")
cur.execute("INSERT INTO entries_fts(rowid, word, pos, definition, synonyms) SELECT id, word, pos, definition, synonyms FROM entries;")
cur.execute("INSERT INTO entries_fts(entries_fts) VALUES('optimize');")
conn.commit()

size_mb = os.path.getsize(db_path) / (1024 * 1024)
print(f"Done! Total entries: {count:,} | DB Size: {size_mb:.2f} MB")
conn.close()
