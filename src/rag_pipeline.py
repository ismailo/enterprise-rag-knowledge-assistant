from openai import OpenAI

client = OpenAI()

def generate_answer(query, context_chunks):

    context = "\n\n".join(context_chunks)

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