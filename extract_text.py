"""Multi-page handwritten answer-sheet OCR using line detection + TrOCR.

Run from the project folder:
  .\\.venv\\Scripts\\python.exe extract_text.py "dataset/Student_Pdf/Student_10.pdf" --detect-only
  .\\.venv\\Scripts\\python.exe extract_text.py "dataset/Student_Pdf/Student_10.pdf"

The first command detects lines and saves debug overlays without loading TrOCR.
Inspect those overlays before running OCR.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import fitz
import numpy as np
from PIL import Image


MODEL_NAME = "microsoft/trocr-base-handwritten"


def render_page(page: fitz.Page, scale: float = 2.0) -> Image.Image:
    """Render a PDF page at 144 DPI for clearer handwriting crops."""
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def get_ink_mask(rgb: np.ndarray) -> np.ndarray:
    """Prefer blue pen; fall back to dark ink when little blue is detected."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    blue = cv2.inRange(
        hsv,
        np.array([85, 35, 25], dtype=np.uint8),
        np.array([150, 255, 245], dtype=np.uint8),
    )

    min_blue_pixels = max(
        200,
        int(rgb.shape[0] * rgb.shape[1] * 0.00008),
    )
    if cv2.countNonZero(blue) >= min_blue_pixels:
        return blue

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        12,
    )


def remove_page_rules(mask: np.ndarray) -> np.ndarray:
    """Remove long straight page rules while retaining nearby handwriting."""
    height, width = mask.shape
    cleaned = mask.copy()

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(80, int(width * 0.35)), 1),
    )
    horizontal_rules = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_OPEN,
        horizontal_kernel,
    )

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, max(80, int(height * 0.25))),
    )
    vertical_rules = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_OPEN,
        vertical_kernel,
    )

    cleaned = cv2.subtract(cleaned, horizontal_rules)
    cleaned = cv2.subtract(cleaned, vertical_rules)
    return cleaned


def group_components_into_lines(
    boxes: list[tuple[int, int, int, int]],
    page_height: int,
) -> list[tuple[int, int, int, int]]:
    """Group word/character boxes by vertical center into text-line boxes."""
    if not boxes:
        return []

    heights = [h for _, _, _, h in boxes if h >= 6]
    if not heights:
        return []

    median_height = float(np.median(heights))
    tolerance = max(9.0, min(median_height * 0.65, page_height * 0.018))

    boxes = sorted(boxes, key=lambda box: (box[1] + box[3] / 2, box[0]))
    groups: list[list[tuple[int, int, int, int]]] = []
    centers: list[float] = []

    for box in boxes:
        x, y, w, h = box
        center = y + h / 2

        best_index = None
        best_distance = float("inf")
        for index, current_center in enumerate(centers):
            distance = abs(center - current_center)
            if distance <= tolerance and distance < best_distance:
                best_index = index
                best_distance = distance

        if best_index is None:
            groups.append([box])
            centers.append(center)
        else:
            groups[best_index].append(box)
            centers[best_index] = float(
                np.median(
                    [item[1] + item[3] / 2 for item in groups[best_index]]
                )
            )

    lines = []
    for group in groups:
        x1 = min(item[0] for item in group)
        y1 = min(item[1] for item in group)
        x2 = max(item[0] + item[2] for item in group)
        y2 = max(item[1] + item[3] for item in group)

        width = x2 - x1
        height = y2 - y1
        if width >= 18 and height >= 8:
            lines.append((x1, y1, width, height))

    return sorted(lines, key=lambda box: (box[1], box[0]))


def detect_lines(
    rgb: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """Detect candidate text lines. Boxes are (x, y, width, height)."""
    mask = get_ink_mask(rgb)
    cleaned = remove_page_rules(mask)
    page_height, page_width = cleaned.shape

    # Join nearby strokes so letters in the same word form connected regions.
    join_width = max(7, int(page_width * 0.012))
    join_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (join_width, 3),
    )
    joined = cv2.dilate(cleaned, join_kernel, iterations=1)

    contours, _ = cv2.findContours(
        joined,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    components = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        if area < 8 or width < 4 or height < 6:
            continue
        if height > page_height * 0.08:
            continue
        if width > page_width * 0.85:
            continue
        if width < 7 and height < 10:
            continue

        components.append((x, y, width, height))

    lines = group_components_into_lines(components, page_height)

    filtered = []
    for x, y, width, height in lines:
        if height > page_height * 0.09:
            continue
        if width > page_width * 0.95:
            continue
        if y > int(page_height * 0.96):
            continue
        filtered.append((x, y, width, height))

    return cleaned, sorted(filtered, key=lambda box: (box[1], box[0]))


def split_long_line(
    line_rgb: np.ndarray,
    ink_mask: np.ndarray,
    max_aspect_ratio: float = 5.0,
) -> list[Image.Image]:
    """Split unusually wide line crops near low-ink gaps for TrOCR."""
    height, width = ink_mask.shape
    if height == 0 or width == 0:
        return []

    max_width = max(180, int(height * max_aspect_ratio))
    pieces = []
    start = 0

    while width - start > max_width:
        low = start + int(max_width * 0.55)
        high = min(width - 1, start + max_width)

        projection = (ink_mask[:, low:high] > 0).sum(axis=0).astype(np.float32)
        if projection.size == 0:
            cut = start + max_width
        else:
            smooth = np.convolve(
                projection,
                np.ones( nine := 9, dtype=np.float32) / nine,
                mode="same",
            )
            cut = low + int(np.argmin(smooth))

        cut = max(start + int(max_width * 0.5), min(cut, start + max_width))
        if cut <= start:
            cut = min(width, start + max_width)

        piece = line_rgb[:, start:cut]
        if piece.size:
            pieces.append(Image.fromarray(piece))
        start = cut

    if start < width:
        piece = line_rgb[:, start:width]
        if piece.size:
            pieces.append(Image.fromarray(piece))

    return pieces


def load_trocr():
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    print(f"Loaded {MODEL_NAME} on {device}")
    return processor, model, device, torch


def recognize_image(image: Image.Image, processor, model, device, torch) -> str:
    pixel_values = processor(
        images=image.convert("RGB"),
        return_tensors="pt",
    ).pixel_values.to(device)

    with torch.inference_mode():
        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=96,
            num_beams=1,
        )

    return processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0].strip()


def recognize_line(
    rgb: np.ndarray,
    box: tuple[int, int, int, int],
    cleaned_mask: np.ndarray,
    processor,
    model,
    device,
    torch,
) -> str:
    x, y, width, height = box
    pad_x = max(6, int(rgb.shape[1] * 0.004))
    pad_y = max(6, int(height * 0.25))

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(rgb.shape[1], x + width + pad_x)
    y2 = min(rgb.shape[0], y + height + pad_y)

    crop_rgb = rgb[y1:y2, x1:x2]
    crop_mask = cleaned_mask[y1:y2, x1:x2]
    if crop_rgb.size == 0:
        return ""

    pieces = split_long_line(crop_rgb, crop_mask)
    recognized = [
        recognize_image(piece, processor, model, device, torch)
        for piece in pieces
    ]
    return " ".join(text for text in recognized if text).strip()


def starts_new_answer(text: str) -> bool:
    return bool(
        re.match(
            r"^\s*(?:q(?:uestion)?\s*)?\d{1,2}\s*[:.)-]\s*",
            text,
            flags=re.IGNORECASE,
        )
    )


def build_paragraphs(page_lines: list[dict]) -> list[str]:
    """Join wrapped lines for a readable preview; raw lines remain in JSON."""
    paragraphs: list[str] = []

    for item in page_lines:
        text = item["text"].strip()
        if not text:
            continue

        if not paragraphs or starts_new_answer(text):
            paragraphs.append(text)
        else:
            paragraphs[-1] += " " + text

    return paragraphs


QUESTION_PREFIX = re.compile(
    r"^\s*[\"'“”]?\s*(?:q(?:uestion)?\s*)?(\d{1,2})\s*"
    r"[.):\-]\s*[)\]]?\s*(.*)$",
    flags=re.IGNORECASE,
)


def clean_ocr_fragment(text: str) -> str:
    """Normalize spacing and discard obvious binary-mark / punctuation noise."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""

    # Strip embedded runs of diagram labels while retaining surrounding prose.
    text = re.sub(r"(?<!\w)(?:[01]\s+){2,}[01](?!\w)", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" \t,;:")
    if not text:
        return ""

    # The line detector often isolates repeated handwritten 0/1 marks.
    # Keep numeric answers, but discard longer alternating mark sequences.
    tokens = re.findall(r"\d+|[A-Za-z]+", text)
    if tokens and all(token in {"0", "1"} for token in tokens) and len(tokens) >= 3:
        return ""
    if not re.search(r"[A-Za-z0-9]", text):
        return ""
    return text


def extract_numbered_answers(pages: list[dict]) -> list[dict]:
    """Join OCR line fragments under question numbers in reading order.

    A question number is accepted only when it has clear punctuation. If OCR
    repeats a number already seen (common for a misread wrapped line), the
    fragment is joined to the current answer instead of creating a duplicate.
    """
    answers: list[dict] = []
    by_number: dict[int, dict] = {}
    current: dict | None = None

    for page in pages:
        for line in page.get("lines", []):
            text = clean_ocr_fragment(line.get("text", ""))
            if not text:
                continue

            match = QUESTION_PREFIX.match(text)
            if match:
                number = int(match.group(1))
                remainder = clean_ocr_fragment(match.group(2))
                if 1 <= number <= 35 and number not in by_number:
                    current = {"question_number": number, "answer": remainder}
                    answers.append(current)
                    by_number[number] = current
                    continue

            # Wrapped lines continue the latest numbered answer. Ignore
            # leading OCR debris before the first legible question number.
            if current is not None:
                current["answer"] = " ".join(
                    part for part in (current["answer"], text) if part
                )

    return answers


def process_pdf(pdf_path, output_dir, processor, model, device, torch, detect_only):
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_pages = []

    with fitz.open(pdf_path) as pdf:
        print(f"Processing {len(pdf)} pages from {pdf_path.name}")

        for page_index, page in enumerate(pdf):
            page_number = page_index + 1
            page_image = render_page(page, scale=2.0)
            rgb = np.asarray(page_image.convert("RGB"))

            cleaned_mask, boxes = detect_lines(rgb)
            ink_pixels = int(np.count_nonzero(cleaned_mask))

            debug = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
            page_lines = []

            for line_number, (x, y, width, height) in enumerate(boxes, start=1):
                cv2.rectangle(
                    debug,
                    (x, y),
                    (x + width, y + height),
                    (0, 220, 0),
                    2,
                )
                cv2.putText(
                    debug,
                    str(line_number),
                    (x, max(18, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )

                text = ""
                if not detect_only:
                    text = recognize_line(
                        rgb,
                        (x, y, width, height),
                        cleaned_mask,
                        processor,
                        model,
                        device,
                        torch,
                    )
                    print(f"Page {page_number}, line {line_number}: {text}")

                page_lines.append(
                    {
                        "page_number": page_number,
                        "line_number": line_number,
                        "bbox_xywh": [int(x), int(y), int(width), int(height)],
                        "text": text,
                    }
                )

            debug_path = output_dir / f"page_{page_number:03d}_debug.png"
            cv2.imwrite(str(debug_path), debug)

            if detect_only:
                status = "detection_only"
                paragraphs = []
                page_text = (
                    f"[Detection only: {len(boxes)} candidate lines; "
                    f"{ink_pixels} ink pixels]"
                )
            elif not boxes:
                status = "no_lines_detected"
                paragraphs = []
                page_text = "[No text lines detected]"
            else:
                status = "processed"
                paragraphs = build_paragraphs(page_lines)
                page_text = "\n".join(paragraphs) if paragraphs else "[No text recognized]"

            all_pages.append(
                {
                    "page_number": page_number,
                    "status": status,
                    "image_size": [int(rgb.shape[1]), int(rgb.shape[0])],
                    "lines": page_lines,
                    "paragraphs": paragraphs,
                    "debug_overlay": debug_path.name,
                }
            )

            print(f"Page {page_number}: {len(boxes)} candidate lines")

    json_path = output_dir / "extracted_text.json"
    text_path = output_dir / "extracted_text.txt"

    answers = extract_numbered_answers(all_pages) if not detect_only else []
    result = {
        "source_pdf": str(pdf_path),
        "model": MODEL_NAME if not detect_only else None,
        "format": "one question and answer per line",
        "notes": (
            "Answers are grouped from OCR lines by detected question number. "
            "Wrapped lines are joined; repeated binary-mark noise is discarded. "
            "Review uncertain handwriting and debug overlays against the PDF."
        ),
        "answers": answers,
    }

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)

    with text_path.open("w", encoding="utf-8") as file:
        file.write("\n".join(
            f"{item['question_number']}. {item['answer']}" for item in answers
        ))
        file.write("\n")

    print("\nFinished:", pdf_path.name)
    print("Text:", text_path.resolve())
    print("JSON:", json_path.resolve())
    print("Debug overlays:", output_dir.resolve())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        nargs="?",
        default=r"dataset/Student_Pdf",
        help="A student PDF or a folder containing student PDFs",
    )
    parser.add_argument(
        "--output",
        default="outputs/ocr",
        help="Output root; folder inputs get one subfolder per PDF",
    )
    parser.add_argument(
        "--detect-only",
        action="store_true",
        help="Detect and draw line boxes without loading TrOCR",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path.resolve()}")
    if input_path.is_dir():
        pdf_paths = sorted(input_path.glob("*.pdf"))
        if not pdf_paths:
            raise FileNotFoundError(f"No PDF files found in: {input_path.resolve()}")
        output_root = Path(args.output)
    elif input_path.suffix.lower() == ".pdf":
        pdf_paths = [input_path]
        output_root = Path(args.output)
    else:
        raise ValueError("Input must be a PDF file or a folder of PDF files")

    processor = model = device = torch = None
    if not args.detect_only:
        processor, model, device, torch = load_trocr()

    for pdf_path in pdf_paths:
        # Separate directories prevent each student's JSON/TXT and page images
        # from overwriting another student's results.
        output_dir = (
            output_root / pdf_path.stem
            if input_path.is_dir()
            else output_root
        )
        process_pdf(
            pdf_path,
            output_dir,
            processor,
            model,
            device,
            torch,
            args.detect_only,
        )


if __name__ == "__main__":
    main()
