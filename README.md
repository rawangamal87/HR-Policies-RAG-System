# Simple RAG System — Document Question Answering

A lightweight Retrieval-Augmented Generation (RAG) system that answers questions using information retrieved from a user-provided PDF document or Excel spreadsheet. The project combines semantic search and a language model to generate context-aware answers grounded in the uploaded knowledge base.

## 1. Project Overview

Large Language Models (LLMs) can generate natural and useful responses, but they may produce incorrect information or hallucinations when answering questions about specific documents they have not seen.

This project addresses that challenge by implementing a simple RAG pipeline. Instead of relying only on the language model's internal knowledge, the system retrieves relevant information from a user-provided document and uses that information as context when generating an answer.

Users can upload a PDF or Excel file, ask questions about its contents, and receive answers based on the retrieved document excerpts.

The project demonstrates the fundamental components of a RAG application:

- Document ingestion and text extraction
- Text chunking and preprocessing
- Semantic embeddings
- Vector similarity search
- Context-grounded answer generation
- Source inspection and basic relevance filtering

## 2. Project Objectives

The main objectives are to:

1. Build a simple end-to-end RAG pipeline.
2. Support PDF and Excel documents as knowledge sources.
3. Convert document content into vector embeddings.
4. Retrieve relevant text chunks based on semantic similarity.
5. Generate answers using a language model and retrieved context.
6. Display source information to help users inspect the evidence behind an answer.
7. Test how the system behaves when relevant information is unavailable.

## 3. System Architecture

The system follows a Retrieval-Augmented Generation architecture.

```text
            file.md
                   |
                   v
           Document Loader
                   |
                   v
            Text Extraction
                   |
                   v
               Chunking
                   |
                   v
        Sentence Transformer Model
                   |
                   v
            Text Embeddings
                   |
                   v
             FAISS Index
                   |
                   v
User Question ---> Query Embedding
                   |
                   v
           Similarity Search
                   |
                   v
          Relevant Text Chunks
                   |
                   v
         Context + User Question
                   |
                   v
           FLAN-T5 Generator
                   |
                   v
            Generated Answer
                   |
                   v
       Answer + Retrieved Sources
```

### How the Architecture Works

**Step 1 — Document Ingestion**

The user uploads a PDF or Excel file. The document loader extracts readable text from PDF pages or converts Excel rows into text passages.

**Step 2 — Text Chunking**

The extracted content is divided into smaller text chunks. Chunking makes retrieval more focused and helps avoid passing an entire large document to the language model.

The implementation uses a configurable chunk size and overlap. Overlapping chunks preserve some surrounding context between adjacent text segments.

**Step 3 — Embedding Generation**

Each chunk is converted into a numerical vector using the `sentence-transformers/all-MiniLM-L6-v2` model.

Embeddings represent semantic characteristics of text, allowing the system to find related passages even when the query and document use different wording.

**Step 4 — Vector Indexing**

The generated embeddings are stored in a FAISS index. The implementation uses `IndexFlatIP`, which performs exact inner-product search. Because the vectors are normalized, the inner product corresponds to cosine similarity.

**Step 5 — Query Processing and Retrieval**

When a user asks a question, the same embedding model converts the question into a vector. FAISS searches for the document chunks with the highest similarity scores.

The system retrieves the top-ranked chunks and applies a configurable minimum similarity threshold.

**Step 6 — Context Construction**

The retrieved text chunks are combined into a context passage. This context is provided alongside the original question to the generator.

**Step 7 — Answer Generation**

The `google/flan-t5-small` model generates a response using the supplied context and instructions to avoid unsupported answers.

**Step 8 — Source Display**

The system displays the generated answer and information about the retrieved sources, including PDF page numbers or Excel sheet and row information when available.

This makes the retrieval process more transparent and helps users verify the answer against the original document.

## 4. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Google Colab | Development and execution environment |
| PyMuPDF | Extracting text from PDF files |
| Pandas | Reading and processing Excel data |
| OpenPyXL | Supporting Excel file processing |
| Sentence Transformers | Generating semantic embeddings |
| FAISS | Efficient vector similarity search |
| Hugging Face Transformers | Loading and running the generation model |
| FLAN-T5 Small | Generating context-based answers |
| NumPy | Numerical array and embedding operations |

The project uses pretrained models rather than training a language model from scratch. The embedding and generation models serve different purposes: one supports retrieval, while the other generates answers.

## 5. Supported Input Formats

### PDF

The loader extracts text from each PDF page and records the corresponding page number.

### Excel

The loader reads Excel sheets using Pandas and converts each non-empty row into a readable text passage. Column names are included so that the meaning of each value is preserved during retrieval.

For example, a spreadsheet row can be converted into text such as:

```text
Employee: Ahmed; Department: IT; Leave Balance: 12
```

This representation makes individual rows searchable using natural-language questions.

**Note:** Standard text extraction may not work for scanned PDFs that contain images instead of selectable text. Such files require an OCR step.

## 6. Core Components

### Document Loader

Responsible for reading the uploaded file, extracting text, and retaining basic source metadata.

### Chunking Function

Splits extracted text into smaller overlapping passages. Chunk size and overlap can be adjusted depending on the document structure.

### Embedding Model

The `all-MiniLM-L6-v2` model converts both document chunks and user questions into vectors in the same embedding space.

### FAISS Retriever

The retriever compares the query embedding with indexed document embeddings and returns the most similar passages.

The current implementation uses exact similarity search, making it straightforward to understand and suitable for a small prototype.

### Generator

The FLAN-T5 Small model receives a prompt containing the retrieved context and user question, then generates a response.

### RAG Orchestration Function

The main question-answering function connects retrieval, relevance filtering, answer generation, and source display into a single workflow.

## 7. Installation and Setup

The project is designed to run in Google Colab.

### Step 1 — Open Google Colab

Visit [Google Colab](https://colab.research.google.com/) and create a new notebook.

### Step 2 — Install Dependencies

Run the following cell:

```python
!pip -q install pymupdf pandas openpyxl sentence-transformers faiss-cpu transformers sentencepiece
```

### Step 3 — Import Libraries

Import the required libraries for document processing, embedding generation, vector search, and text generation.

### Step 4 — Upload a Document

Upload a PDF or Excel file using the Colab file-upload interface.

### Step 5 — Execute the Pipeline

Run the notebook cells in order to extract text, create chunks, generate embeddings, build the FAISS index, load the generator, and define the question-answering function.

### Step 6 — Ask Questions

Call the RAG function with a question relevant to the uploaded file:

```python
ask_question("What is the main topic of this document?")
```

Replace the example question with a question that can be answered using the uploaded document.

The first run may take additional time because pretrained models need to be downloaded.

## 8. Example Workflow

Suppose the user uploads an employee policy document containing leave policies, attendance rules, and employee benefits.

The user asks:

```text
How many annual leave days are provided to employees?
```

The system:

1. Converts the question into an embedding.
2. Retrieves the most relevant policy passages from FAISS.
3. Combines the passages with the question.
4. Sends the context to FLAN-T5.
5. Displays the generated answer and the retrieved source excerpts.

The actual answer depends on the uploaded document. The system should not assume a policy value before retrieving supporting evidence.

## 9. Handling Missing Information

The system includes a basic similarity threshold to reject retrieval results that appear insufficiently relevant.

When no chunks pass this threshold, it returns a message indicating that relevant information could not be found.

The generation prompt also instructs the model to state when the answer is not supported by the provided context.

However, these safeguards do not guarantee that every generated answer is correct. Similarity thresholds are heuristic, and a language model can still generate unsupported statements. The retrieved excerpts should therefore be checked during testing.

## 10. Testing and Validation

The system should be tested using several categories of questions:

| Test Case | Expected Behavior |
|---|---|
| A question explicitly answered in the document | Retrieve relevant evidence and generate an appropriate answer |
| A question expressed using different wording | Retrieve semantically related passages when possible |
| A question about a specific PDF page or Excel row | Return relevant content and source metadata |
| A question unrelated to the document | Reject low-relevance results when the threshold permits |
| An empty or unreadable document | Report that no usable text was extracted |
| A PDF containing scanned pages | Require OCR if ordinary text extraction returns no content |

Retrieval quality and answer quality should be evaluated separately. A correct answer requires both relevant retrieved evidence and an appropriate generation step.

## 11. Current Limitations

This implementation is intentionally simple and designed for educational purposes.

- **Small generation model:** FLAN-T5 Small may struggle with complex questions, long contexts, or detailed reasoning.
- **Context length:** The prompt is truncated to fit the model's configured input length. Relevant evidence near the end of a long context may be lost.
- **Heuristic relevance threshold:** A fixed similarity threshold may reject useful passages or accept irrelevant ones.
- **PDF extraction:** Complex layouts, tables, and scanned documents may not be extracted accurately.
- **Excel processing:** Row-based text conversion may lose relationships across multiple rows or complex worksheets.
- **In-memory vector index:** The FAISS index is created during the current notebook session and is not automatically persisted.
- **No user interface:** The current version uses Google Colab input cells rather than a standalone web application.
- **No formal evaluation dataset:** The prototype does not yet calculate quantitative retrieval or answer-generation metrics.

These limitations provide opportunities for future improvement.

## 12. Future Improvements

Potential enhancements include:

1. Adding OCR support for scanned PDFs.
2. Using stronger embedding and generation models.
3. Improving chunking with document-aware splitting.
4. Adding a reranking model to improve retrieval precision.
5. Saving and reloading the FAISS index.
6. Building a Streamlit or FastAPI interface.
7. Adding answer citations with exact source excerpts.
8. Evaluating retrieval with Precision@K, Recall@K, and Mean Reciprocal Rank.
9. Evaluating answer correctness and groundedness using a test dataset.
10. Supporting multiple documents in a persistent knowledge base.

## 13. Key Concepts Demonstrated

This project provides hands-on experience with:

- Retrieval-Augmented Generation (RAG)
- Document ingestion and preprocessing
- Text chunking and overlapping windows
- Semantic embeddings
- Vector databases and similarity search
- Query embedding and retrieval
- Prompt construction and context grounding
- Language model inference
- Basic source tracking
- Hallucination mitigation and system limitations

## 14. Conclusion

The Simple RAG System demonstrates how document-based question answering can be implemented by combining semantic retrieval with language model generation.

By extracting content from PDF or Excel files, converting the content into embeddings, retrieving relevant passages with FAISS, and passing those passages to FLAN-T5, the system creates an end-to-end workflow for answering questions about a private knowledge source.

The project serves as a foundation for more advanced document assistants, internal knowledge-base search systems, and enterprise RAG applications.

**Project Type:** AI Engineering / Natural Language Processing / Retrieval-Augmented Generation

**Environment:** Google Colab

**Primary Goal:** Answer questions using information retrieved from a user-provided document.
