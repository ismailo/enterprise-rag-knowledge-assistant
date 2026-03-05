from pathlib import Path
from typing import List, Dict, Any
from pypdf import PdfReader


def load_documents(folder_path: str) -> List[Dict[str, Any]]:
    """
    Loads .pdf and .txt files from folder_path.
    Returns list of {source, pages:[{page_num, text}], full_text}.
    """
    docs: List[Dict[str, Any]] = []
    folder = Path(folder_path)

    # PDFs
    for file in folder.glob("*.pdf"):
        reader = PdfReader(str(file))
        pages = []
        full_text = ""

        for idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append({"page_num": idx + 1, "text": text})
            full_text += text + "\n"

        docs.append({"source": file.name, "pages": pages, "full_text": full_text})

    # TXTs
    for file in folder.glob("*.txt"):
        text = file.read_text(encoding="utf-8", errors="ignore")
        docs.append({
            "source": file.name,
            "pages": [{"page_num": 1, "text": text}],
            "full_text": text
        })

    return docs