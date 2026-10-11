
from pathlib import Path
from time import perf_counter
from paddleocr import PaddleOCR

PDF = Path(r"Student_Pdf\Student_1.pdf")
OUT = Path(r"outputs\fast_ocr")

OUT.mkdir(parents=True, exist_ok=True)

if not PDF.exists():
    raise FileNotFoundError(f"PDF not found: {PDF}")

print("Loading lightweight English OCR model...")
start = perf_counter()

ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="en_PP-OCRv5_mobile_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)

print(f"Model initialization: {perf_counter() - start:.1f}s")

# Render one page at a time to avoid processing the whole PDF
# at once. PyMuPDF is imported using its current module name.
import pymupdf

doc = pymupdf.open(str(PDF))

all_text = []

try:
    for page_index, page in enumerate(doc):
        print(f"\nProcessing page {page_index + 1}/{len(doc)}...")
        start = perf_counter()

        # Render at a moderate resolution for a speed test.
        pix = page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False)
        image_path = OUT / f"page_{page_index + 1:02d}.png"
        pix.save(str(image_path))

        results = ocr.predict(input=str(image_path))
        page_lines = [f"\n--- PAGE {page_index + 1} ---\n"]

        for result in results:
            data = result.json
            if isinstance(data, dict):
                data = data.get("res", data)

            if not isinstance(data, dict):
                continue

            texts = data.get("rec_texts", [])
            scores = data.get("rec_scores", [])

            for i, text in enumerate(texts):
                score = scores[i] if i < len(scores) else None
                page_lines.append(
                    f"{text} [confidence={score:.3f}]"
                    if isinstance(score, (float, int))
                    else str(text)
                )

        all_text.extend(page_lines)
        print("\n".join(page_lines))
        print(f"Page time: {perf_counter() - start:.1f}s")

finally:
    doc.close()

text_file = OUT / "extracted_text.txt"
text_file.write_text("\n".join(all_text), encoding="utf-8")

print(f"\nFinished. Text saved to: {text_file}")
