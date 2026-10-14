
from retriever import Retriever
from llm import HRPolicyLLM
from vector_store import VectorStore

# Connect the retriever and language model so retrieved policy information can be used to generate answers to employee questions.
class RAGPipeline:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
# retrieve relevant chunks, build a context, generate an answer, and return the answer together with the related source filenames.
    def answer(self, question):
        if not question.strip():
            return {
                "question": question,
                "answer": "Please enter a question.",
                "sources": [],
            }

        results = self.retriever.retrieve(question)

        if not results:
            return {
                "question": question,
                "answer": (
                    "I couldn't find relevant information "
                    "in the available HR policies."
                ),
                "sources": [],
            }

        context = "\n\n".join(
            f"Source: {result['source']}\n{result['text']}"
            for result in results
        )
# Generate an answer using the employee's question and the retrieved HR policy context.
        answer = self.llm.generate_answer(
            question=question,
            context=context,
        )

        sources = list(dict.fromkeys(
            result["source"] for result in results
        ))
        # Return the original question, generated answer, and source list.

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
        }


if __name__ == "__main__":
    vector_store = VectorStore()
    retriever = Retriever(vector_store, top_k=3)
    llm = HRPolicyLLM(model_name="llama3.1:latest")

    rag = RAGPipeline(retriever, llm)

    questions = [
        "How do I request vacation?",
        "How much notice is required for a two-week vacation?",
        "What is the company's remote work policy?",
    ]

    for question in questions:
        result = rag.answer(question)

        print("\n" + "=" * 60)
        print("Question:", result["question"])
        print("\nAnswer:", result["answer"])
        print("\nSources:")
        for source in result["sources"]:
            print("-", source)
