from PyPDF2 import PdfReader
def pdf_to_txt(pdf_path, txt_path):
    reader = PdfReader(pdf_path)
    with open(txt_path, "w", encoding="utf-8") as f:
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                f.write(f"--- Страница {i+1} ---\n")
                f.write(text)
                f.write("\n\n")

pdf_to_txt("test.pdf", "output.txt")
