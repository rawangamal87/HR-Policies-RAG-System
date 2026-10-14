

import ollama

class HRPolicyLLM:
    def __init__(self, model_name="llama3.1:latest"):
        self.model_name = model_name

    def generate_answer(self, question, context):
        prompt = f"""
You are an HR Policy Assistant.

Answer the employee's question using ONLY
the HR policy context provided below.

Rules:
- Do not invent company policies.
- If the context does not contain the answer,
  clearly say that the information was not found.
- Give a clear, concise answer.
- Mention the relevant policy details.

HR POLICY CONTEXT:
{context}

EMPLOYEE QUESTION:
{question}

ANSWER:
"""

        response = ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            options={"temperature": 0}
        )

        return response["message"]["content"]
