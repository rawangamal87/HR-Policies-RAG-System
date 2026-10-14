from typing import List
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

# Load the Sentence Transformer model used to convert text into numerical vectors that represent semantic meaning.
class EmbeddingModel:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.model = SentenceTransformer(model_name)
# Convert a single text into a normalized embedding vector and return the vector as a Python list.
    def embed_text(self, text: str) -> List[float]:
        embedding = self.model.encode(
            text,
            normalize_embeddings=True
        )
        return embedding.tolist()
    # Convert multiple text chunks into embedding vectors in one batch to prepare the documents for storage in the vector database.
    def embed_documents(
        self,
        texts: List[str]
    ) -> List[List[float]]:
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        return self.embed_text(query)


if __name__ == "__main__":
    embedding_model = EmbeddingModel()

    text = "Employees can request annual leave."
    vector = embedding_model.embed_text(text)

    print("Embedding created successfully!")
    print(f"Vector dimensions: {len(vector)}")
    print(f"First 5 values: {vector[:5]}")


