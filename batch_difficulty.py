import os
import re
import subprocess
import fitz
import numpy as np
import pandas as pd

FORMAL_TERMS = ["Definition", "Theorem", "Proof", "Lemma", "Assume", "Let"]
CROSS_REF_TERMS = ["Recall", "As shown", "From Chapter", "We previously", "Earlier we"]

def ocr_pdf_if_needed(pdf_path: str) -> str:
    base, ext = os.path.splitext(pdf_path)
    ocr_path = f"{base}_OCR{ext}"

    if os.path.exists(ocr_path):
        return ocr_path

    try:
        doc = fitz.open(pdf_path)
        sample_text = doc.load_page(0).get_text("text")
        if len(sample_text.strip()) < 50:
            print(f"OCR running for {os.path.basename(pdf_path)}...")
            subprocess.run(
                ["ocrmypdf", "--skip-text", pdf_path, ocr_path],
                check=True
            )
            return ocr_path
    except Exception:
        pass

    return pdf_path

def symbol_density(text: str) -> float:
    if not text:
        return 0
    symbols = re.findall(r"[^a-zA-Z0-9\s]", text)
    return len(symbols) / max(len(text), 1)

def count_terms(text: str, terms: list[str]) -> int:
    return sum(text.count(t) for t in terms)

def analyze_book(pdf_path: str) -> dict:
    pdf_path = ocr_pdf_if_needed(pdf_path)

    doc = fitz.open(pdf_path)
    total_chars = 0
    total_symbols = 0
    total_formal = 0
    total_cross = 0

    for i in range(doc.page_count):
        text = doc.load_page(i).get_text("text")
        if not text:
            continue

        total_chars += len(text)
        total_symbols += len(re.findall(r"[^a-zA-Z0-9\s]", text))
        total_formal += count_terms(text, FORMAL_TERMS)
        total_cross += count_terms(text, CROSS_REF_TERMS)

    if total_chars == 0:
        return None

    sym_density = total_symbols / total_chars
    formal_density = total_formal / doc.page_count
    cross_density = total_cross / doc.page_count

    difficulty_score = (
        5 * sym_density
        + 2 * formal_density
        + 1.5 * cross_density
    )

    return {
        "book": os.path.basename(pdf_path),
        "pages": doc.page_count,
        "symbol_density": sym_density,
        "formal_density": formal_density,
        "cross_reference_density": cross_density,
        "difficulty_score": difficulty_score,
    }

def analyze_folder(folder_path: str, output_csv="book_difficulty_ranking.csv"):
    results = []

    for file in os.listdir(folder_path):
        if file.lower().endswith(".pdf"):
            full_path = os.path.join(folder_path, file)
            print(f"Analyzing {file}...")
            result = analyze_book(full_path)
            if result:
                results.append(result)

    df = pd.DataFrame(results)
    df = df.sort_values("difficulty_score", ascending=False)

    df.to_csv(output_csv, index=False)
    print(f"\nSaved ranking to {output_csv}")
    print(df)

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python batch_difficulty.py <folder_path>")
        raise SystemExit(1)

    folder = sys.argv[1]
    analyze_folder(folder)