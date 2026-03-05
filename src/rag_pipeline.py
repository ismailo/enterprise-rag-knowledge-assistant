import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_answer(query: str, context_chunks: List[Dict[str, Any]]) -> str:
    """
    Generates answer grounded in context. Forces citations.
    """
    # Build context block with labels
    context_blocks = []
    for i, c in enumerate(context_chunks, start=1):
        label = f"[S{i}] {c['source']} (p.{c['page_num']}, chunk {c['chunk_id']})"
        context_blocks.append(f"{label}\n{c['text']}")
    context = "\n\n".join(context_blocks)

    prompt = f"""You are a careful assistant.
Answer the question using ONLY the sources provided.
If the sources do not contain the answer, say: "I don't know based on the provided documents."

When you make a claim, cite sources like [S1], [S2]. Do not cite anything else.

Question:
{query}

Sources:
{context}
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()