import fitz  # PyMuPDF

pdfs = [
        "data/PRODUCT DATA - 2245.pdf",
        "data/PRODUCT DATA - 2250 2270.pdf",
        "data/PRODUCT DATA - 2255.pdf",
        "data/PRODUCT DATA - 2270.pdf",
        "data/PRODUCT DATA - 3668.pdf",
        "data/PRODUCT DATA - Dirac SW.pdf",
        "data/PRODUCT DATA - Noise Partner.pdf"
]

for pdf_path in pdfs:
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text() + "\n"

    # Filter lines with 5 or more words
    filtered_lines = []
    for line in text.split('\n'):
        word_count = len(line.split())
        if word_count >= 2:
            filtered_lines.append(line)
    
    filtered_text = '\n'.join(filtered_lines)

    output_txt_path = pdf_path.replace(".pdf", ".txt").replace("data/", "output_")
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(filtered_text)