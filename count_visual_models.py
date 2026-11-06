import os
import re
import fitz  # PyMuPDF
import pandas as pd

CAPTION_RE = re.compile(
    r"(?mi)^\s*(fig(?:ure)?|diagram|graph|table|chart|model)\s*\.?\s*(\d+(?:[.\-]\d+)*)\b"
)

def count_visual_models_in_pdf(pdf_path: str) -> dict:
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count

    MIN_IMG_FRAC = 0.02 
    MAX_IMG_FRAC = 0.60  
    CAP_PER_PAGE = 5   

    images_total = 0
    captions_total = 0
    pages_with_images = 0
    pages_with_captions = 0

    for i in range(total_pages):
        page = doc.load_page(i)
        page_area = page.rect.width * page.rect.height

        page_xrefs = set()
        for img in page.get_images(full=True):
            xref = img[0]
            for r in page.get_image_rects(xref):
                frac = (r.width * r.height) / max(1.0, page_area)
                if MIN_IMG_FRAC <= frac <= MAX_IMG_FRAC:
                    page_xrefs.add(xref)

        img_count = len(page_xrefs)
        images_total += img_count
        if img_count > 0:
            pages_with_images += 1

        # ---- Count UNIQUE caption labels per page ----
        text = page.get_text("text") or ""
        hits = CAPTION_RE.findall(text)
        unique_caps = {(h[0].lower(), h[1]) for h in hits}
        cap_count = min(len(unique_caps), CAP_PER_PAGE)

        captions_total += cap_count
        if cap_count > 0:
            pages_with_captions += 1

    doc.close()

    visual_models_total = images_total + captions_total

    return {
        "file": os.path.basename(pdf_path),
        "path": pdf_path,
        "pages": total_pages,
        "images_total": images_total,
        "captions_total": captions_total,
        "visual_models_total": visual_models_total,
        "pages_with_images": pages_with_images,
        "pages_with_captions": pages_with_captions,
        "visual_models_per_page": visual_models_total / max(1, total_pages),
    }

def analyze_folder(folder_path: str, output_csv: str = "visual_models_by_book.csv"):
    pdfs = []
    for root, _, files in os.walk(folder_path):
        for name in files:
            if name.lower().endswith(".pdf"):
                pdfs.append(os.path.join(root, name))

    print(f"Found {len(pdfs)} PDFs under: {folder_path}")
    if not pdfs:
        print("No PDFs found. Check the folder path.")
        return

    rows = []
    for pdf_path in sorted(pdfs):
        print("Analyzing:", os.path.basename(pdf_path))
        try:
            rows.append(count_visual_models_in_pdf(pdf_path))
        except Exception as e:
            rows.append({
                "file": os.path.basename(pdf_path),
                "path": pdf_path,
                "error": repr(e)
            })

    df = pd.DataFrame(rows)
    if "visual_models_per_page" in df.columns:
        df = df.sort_values("visual_models_per_page", ascending=False)

    df.to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv}")
    print(df.head(10))

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python count_visual_models.py <folder_path> [output_csv]")
        raise SystemExit(1)

    folder = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) >= 3 else "visual_models_by_book.csv"
    analyze_folder(folder, out)