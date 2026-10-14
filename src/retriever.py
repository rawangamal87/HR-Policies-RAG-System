
from typing import List, Dict
from vector_store import VectorStore

# Initialize the retriever with a vector store and the number of relevant results to return for each query.
class Retriever:
    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int = 3
    ):
        self.vector_store = vector_store
        self.top_k = top_k
 #Validate the user's query, request the most relevant chunks from the vector store, and return the retrieved results.
    def retrieve(self, query: str) -> List[Dict]:
        if not query.strip():
            return []

        results = self.vector_store.search(
            query=query,
            top_k=self.top_k
        )

        return results
