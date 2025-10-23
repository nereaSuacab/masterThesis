import fitz  # PyMuPDF

pdfs = [
        "data/UserGuide - Building Acoustics Partner.pdf",
        "data/UserGuide - Noise Partner.pdf",
        "data/UserGuide-Enviro noise Partner.pdf"
]

for pdf_path in pdfs:
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text() + "\n"

    output_txt_path = pdf_path.replace(".pdf", ".txt").replace("data/", "output_")
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(text)