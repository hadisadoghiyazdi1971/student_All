
"""
Build helper for Ghafarian Chapters 3-4 English LaTeX.

What it does:
1. Reads Chapter_3_4_English.tex if present; otherwise downloads it from GitHub.
2. Removes the original document preamble/end and creates Chapter_3_4_English_patched_body.tex.
3. Normalizes figure paths so all images are expected under figures/.
4. Optionally downloads the original PDF and extracts all embedded images to extracted_pdf_images/.

Run:
    python build_complete_latex.py
Then compile:
    pdflatex Ghafarian_Ch3_4_English_complete.tex
    pdflatex Ghafarian_Ch3_4_English_complete.tex
"""
from pathlib import Path
import re, urllib.request, shutil, os

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "Chapter_3_4_English.tex"
BODY = ROOT / "Chapter_3_4_English_patched_body.tex"
FIGDIR = ROOT / "figures"
PDF = ROOT / "GhafarianThesis_ActiveLearning.pdf"
EXTRACT = ROOT / "extracted_pdf_images"

TEX_URL = "https://raw.githubusercontent.com/hadisadoghiyazdi1971/student_All/refs/heads/main/ghaffarian/Chapter_3_4_English.tex"
PDF_URL = "https://github.com/hadisadoghiyazdi1971/student_All/raw/refs/heads/main/ghaffarian/GhafarianThesis_ActiveLearning.pdf"

EXPECTED = [
    "fig3_1a.png", "fig3_1b.png", "fig3_1c.png", "fig3_1d.png", "fig3_5.png",
    "fig4_1.png", "fig4_2.png", "fig4_3.png", "fig4_4.png", "fig4_5.png",
    "fig4_6.png", "fig4_7.png", "fig4_8.png", "fig4_9.png", "fig4_10.png",
]

def download(url, path):
    print(f"Downloading {url}")
    with urllib.request.urlopen(url, timeout=60) as r:
        path.write_bytes(r.read())

def patch_tex():
    if not SRC.exists():
        download(TEX_URL, SRC)
    s = SRC.read_text(encoding='utf-8', errors='replace')
    s = s.replace('\\documentclass[12pt]{article}', '')
    # Drop the original preamble up to \begin{document}
    s = re.sub(r".*?\\begin\{document\}", "", s, count=1, flags=re.S)
    s = re.sub(r"\\end\{document\}\s*$", "", s, count=1, flags=re.S)
    s = s.replace('\\maketitle', '')
    # The external file uses article + \chapter inconsistently. The wrapper is report.
    s = s.replace('\\section{Active Learning on Distributions}', '\\chapter{Active Learning on Distributions}')
    s = s.replace('<=', r'\\le ')
    # Normalize figure floats and image paths.
    s = s.replace('\\begin{figure}[h!]', '\\begin{figure}[H]')
    s = s.replace('\\begin{figure}[H]\n\t\t\\centering\n\t\t\\includegraphics[width=0.8\\textwidth]{fig4_1.png}',
                  '\\clearpage\n\t\\begin{figure}[H]\n\t\t\\centering\n\t\t\\includegraphics[width=0.8\\textwidth]{fig4_1.png}')
    for name in EXPECTED:
        s = s.replace('{' + name + '}', '{' + name + '}')
    BODY.write_text(s.strip() + "\n", encoding='utf-8')
    print(f"Wrote {BODY}")

def extract_pdf_images():
    try:
        import fitz  # PyMuPDF
    except Exception:
        print('PyMuPDF is not installed. Install with: pip install pymupdf')
        return
    if not PDF.exists():
        try:
            download(PDF_URL, PDF)
        except Exception as e:
            print('Could not download PDF automatically:', e)
            print('Put GhafarianThesis_ActiveLearning.pdf in this folder and rerun.')
            return
    EXTRACT.mkdir(exist_ok=True)
    doc = fitz.open(PDF)
    count = 0
    for pno in range(len(doc)):
        page = doc[pno]
        for imgno, img in enumerate(page.get_images(full=True), start=1):
            xref = img[0]
            data = doc.extract_image(xref)
            ext = data.get('ext', 'png')
            out = EXTRACT / f"page_{pno+1:03d}_img_{imgno:02d}.{ext}"
            out.write_bytes(data['image'])
            count += 1
    print(f"Extracted {count} embedded images to {EXTRACT}")
    print('Now visually map the extracted original images to figures/fig3_1a.png ... figures/fig4_10.png.')

def check_figures():
    missing = [x for x in EXPECTED if not (FIGDIR/x).exists()]
    if missing:
        print('Missing figure files:', missing)
    else:
        print('All expected figure filenames exist in figures/.')

if __name__ == '__main__':
    FIGDIR.mkdir(exist_ok=True)
    patch_tex()
    check_figures()
    extract_pdf_images()
