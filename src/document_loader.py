from pathlib import Path
from pypdf import PdfReader

def load_documents(folder_path):
    documents = []

    for file in Path(folder_path).glob("*.pdf"):
        reader = PdfReader(file)
        text = ""

        for page in reader.pages:
            text += page.extract_text()

        documents.append({
            "source": file.name,
            "text": text
        })

    return documents