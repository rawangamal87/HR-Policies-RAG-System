import glob
from typing import List
from langchain_core.documents import Document


class DocumentLoader:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

        # Search for all Markdown files, read their content, and convert them
    def load_documents(self) -> List[Document]:
        documents = []
        for file_path in glob.glob(f"{self.data_dir}/**/*.md", recursive=True):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                documents.append(Document(page_content=content, metadata={"source": file_path}))
        return documents

if __name__ == "__main__":
    loader = DocumentLoader("Data")
    documents = loader.load_documents()
    print(f"Loaded {len(documents)} documents")



