
import sys
from pathlib import Path

# Add the src directory to Python's import path
SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

from vector_store import VectorStore

vector_store = VectorStore()

print("Stored chunks:", vector_store.collection.count())

results = vector_store.search("annual leave", top_k=3)

print("Retrieved results:", len(results))

for result in results:
    print("\nSource:", result["source"])
    print("Distance:", result["distance"])
    print("Text:", result["text"][:300])
