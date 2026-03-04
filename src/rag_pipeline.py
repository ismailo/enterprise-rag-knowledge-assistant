import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

from openai import OpenAI

client = OpenAI()

def generate_answer(query, context_chunks):

    context = "\n\n".join(chunk["text"] for chunk in context_chunks)

    prompt = f"""
Answer the question using the context below.

Context:
{context}

Question:
{query}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content