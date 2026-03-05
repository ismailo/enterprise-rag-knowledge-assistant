import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def grounding_check(query: str, answer: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Verify whether the answer is grounded in the provided sources.
    """

    source_blocks = []

    for i, s in enumerate(sources, start=1):
        block = f"[S{i}] {s['source']} page {s['page_num']} chunk {s['chunk_id']}\n{s['text']}"
        source_blocks.append(block)

    context = "\n\n".join(source_blocks)

    prompt = f"""
You are a strict evaluator.

Determine if the ANSWER is supported by the SOURCES.

Return ONLY valid JSON with keys:
- supported (true or false)
- unsupported_claims (list)
- notes (short explanation)

QUESTION:
{query}

ANSWER:
{answer}

SOURCES:
{context}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    content = response.choices[0].message.content.strip()

    try:
        return json.loads(content)
    except Exception:
        return {
            "supported": False,
            "unsupported_claims": [],
            "notes": "Could not parse evaluator output"
        }