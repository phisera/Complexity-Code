import re
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pdfplumber
from pypdf import PdfReader
import os
import subprocess
import fitz
import numpy as np
import pandas as pd

def ocr_pdf_if_needed(pdf_path: str) -> str:
    base, ext = os.path.splitext(pdf_path)
    ocr_path = f"{base}_OCR{ext}"

    if os.path.exists(ocr_path):
        return ocr_path

    try:
        pages = extract_pages_multi(pdf_path, min_chars=5)
        non_empty = sum(1 for p in pages[:10] if len(p["text"]) > 20)

        if non_empty <= 2:
            print("Text extraction empty. Running OCR")

            env = os.environ.copy()
            env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:" + env.get("PATH", "")

            subprocess.run(
                ["ocrmypdf", "--skip-text", pdf_path, ocr_path],
                check=True,
                env=env,
            )

            print(f"OCR complete: {ocr_path}")
            return ocr_path

    except Exception as e:
        print("OCR failed", repr(e))

    return pdf_path

def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def extract_pages_multi(pdf_path: str, min_chars: int = 20):
    # PyMuPDF
    mupdf_text = []
    try:
        doc = fitz.open(pdf_path)
        for i in range(doc.page_count):
            t = clean_text(doc.load_page(i).get_text("text"))
            mupdf_text.append(t)
    except Exception:
        mupdf_text = []

    # pdfplumber
    plumber_text = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                plumber_text.append(clean_text(page.extract_text()))
    except Exception:
        plumber_text = []

    # pypdf
    pypdf_text = []
    try:
        reader = PdfReader(pdf_path)
        for p in reader.pages:
            pypdf_text.append(clean_text(p.extract_text()))
    except Exception:
        pypdf_text = []

    n = max(len(mupdf_text), len(plumber_text), len(pypdf_text))
    pages = []
    for i in range(n):
        candidates = []
        if i < len(mupdf_text): candidates.append(mupdf_text[i])
        if i < len(plumber_text): candidates.append(plumber_text[i])
        if i < len(pypdf_text): candidates.append(pypdf_text[i])
        best = max(candidates, key=lambda x: len(x)) if candidates else ""
        pages.append({"page": i+1, "text": best if len(best) >= min_chars else ""})
    return pages

def extract_pages(pdf_path: str, min_chars: int = 20) -> list[dict]:
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(doc.page_count):
        text = doc.load_page(i).get_text("text")
        text = clean_text(text)
        if len(text) >= min_chars:
            pages.append({"page": i + 1, "text": text})
        else:
            pages.append({"page": i + 1, "text": ""})
    return pages

def topic_vectors(pages: list[dict], model_name="all-MiniLM-L6-v2") -> np.ndarray:
    model = SentenceTransformer(model_name)

    texts = [p["text"] for p in pages]
    safe_texts = [t if t else "[EMPTY_PAGE]" for t in texts]

    vecs = model.encode(safe_texts, normalize_embeddings=True)
    for idx, t in enumerate(texts):
        if not t:
            vecs[idx, :] = np.nan
    return vecs

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    sim = cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0, 0]
    return float(1.0 - sim)


def compute_topic_change(pages: list[dict], vecs: np.ndarray) -> pd.DataFrame:
    rows = []
    last_valid_idx = None

    for i in range(len(pages)):
        page_num = pages[i]["page"]
        text_len = len(pages[i]["text"])

        if np.isnan(vecs[i]).any():
            rows.append({
                "page": page_num,
                "text_len": text_len,
                "topic_change": np.nan,
                "compared_to_page": np.nan
            })
            continue

        if last_valid_idx is None:
            rows.append({
                "page": page_num,
                "text_len": text_len,
                "topic_change": 0.0,
                "compared_to_page": np.nan
            })
        else:
            dist = cosine_distance(vecs[last_valid_idx], vecs[i])
            rows.append({
                "page": page_num,
                "text_len": text_len,
                "topic_change": dist,
                "compared_to_page": pages[last_valid_idx]["page"]
            })

        last_valid_idx = i

    df = pd.DataFrame(rows)
    return df

def main(pdf_path: str, out_csv: str = "topic_change.csv"):
    pdf_path = ocr_pdf_if_needed(pdf_path)

    pages = extract_pages_multi(pdf_path)
    vecs = topic_vectors(pages)
    df = compute_topic_change(pages, vecs)

    df["topic_change_smooth_5"] = df["topic_change"].rolling(5, min_periods=1).mean()

    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    print(df.head(10))

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python topic_change.py <pdf_path> [out_csv]")
        raise SystemExit(1)

    pdf = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) >= 3 else "topic_change.csv"
    main(pdf, out)

# for copy-pasting: python3 topic_change.py "insert_textbook.pdf"