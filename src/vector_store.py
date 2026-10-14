
import chromadb
from typing import List
from langchain_core.documents import Document

from embedding_model import EmbeddingModel

# Initialize the persistent ChromaDB database and open or create a collection for HR policies. Reuse an existing embedding model

class VectorStore:
    def __init__(
        self,
        persist_directory: str = "chroma_db",
        collection_name: str = "hr_policies",
        embedding_model=None,
    ):
        # Initialize the embedding model
        if embedding_model is not None:
            self.embedding_model = embedding_model
        else:
            self.embedding_model = EmbeddingModel()

        # Connect to a persistent ChromaDB database
        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        # Create the collection if it does not exist
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, chunks: List[Document]) -> int:
        # Handle an empty list
        if not chunks:
            return 0

        # Extract text from each chunk
        texts = [
            chunk.page_content for chunk in chunks
        ]

        # Convert text chunks into embeddings
        embeddings = self.embedding_model.embed_documents(
            texts
        )

        ids = []
        metadatas = []

        # Prepare IDs and metadata
        for index, chunk in enumerate(chunks):
            source = chunk.metadata.get(
                "source", "unknown"
            )

            ids.append(f"{source}::chunk_{index}")

            metadata = dict(chunk.metadata)
            metadata["chunk_index"] = index
            metadatas.append(metadata)

        # Store or update chunks in ChromaDB
        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return len(chunks)

    def search(self, query: str, top_k: int = 3) -> list:
        # Validate the query and number of results
        if not query.strip() or top_k <= 0:
            return []

        # Check whether the database contains data
        collection_count = self.collection.count()

        if collection_count == 0:
            return []

        # Convert the question into an embedding
        query_embedding = self.embedding_model.embed_query(
            query
        )

        # Retrieve the most relevant chunks
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection_count),
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        matches = []

        # Organize the retrieved results
        for i, text in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i]

            matches.append(
                {
                    "text": text,
                    "source": metadata.get(
                        "source", "unknown"
                    ),
                    "distance": distance,
                }
            )

        return matches


if __name__ == "__main__":
    from document_loader import DocumentLoader
    from chunker import DocumentChunker

    # Step 1: Load HR policy documents
    loader = DocumentLoader("data")
    documents = loader.load_documents()

    print(f"Loaded documents: {len(documents)}")

    # Step 2: Split documents into chunks
    chunker = DocumentChunker()
    chunks = chunker.split_documents(documents)

    print(f"Created chunks: {len(chunks)}")

    # Step 3: Initialize the embedding model
    embedding_model = EmbeddingModel()

    # Step 4: Initialize the vector store
    vector_store = VectorStore(
        embedding_model=embedding_model
    )

    # Step 5: Store the chunks
    stored_count = vector_store.add_documents(chunks)

    print(f"Chunks added or updated: {stored_count}")

    # Step 6: Search for relevant policy chunks
    results = vector_store.search(
        "How can I request annual leave?",
        top_k=3,
    )

    print("\nSearch results:")

    if not results:
        print("No results found.")

    for result in results:
        print(f"\nSource: {result['source']}")
        print(f"Distance: {result['distance']:.4f}")
        print(f"Text: {result['text'][:300]}")
