"""Index student answers and score them against the answer key.

Run from the project folder:
    python semantic_index.py

Run this after the extracted_text.json files are available for all students.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import chromadb
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
OCR_DIR = ROOT / "outputs" / "ocr"
DB_DIR = ROOT / "outputs" / "chroma_db"
SUMMARY_PATH = ROOT / "outputs" / "semantic_scores_summary.json"

QUESTION_FILE = DATASET_DIR / "Question.txt"
ANSWER_KEY_FILE = DATASET_DIR / "answerkey.txt"

QUESTION_COLLECTION = "question_rubrics"
STUDENT_COLLECTION = "student_answers"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EXPECTED_STUDENTS = range(1, 11)

QUESTION_BLOCK = re.compile(
    r"(?ms)^\s*(\d{1,2})\.\s+(.*?)(?=^\s*\d{1,2}\.\s+|\Z)"
)
OPTION_LINE = re.compile(r"^\s*([A-D])\.\s*(.*?)\s*$", re.I | re.M)


def load_text(path: Path) -> str:
    """Load a UTF-8 text file using LangChain's TextLoader."""
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    documents = TextLoader(str(path), encoding="utf-8").load()
    return "\n".join(document.page_content for document in documents)


def parse_questions(text: str) -> dict[int, str]:
    """Split the exam text into numbered question blocks."""
    result = {}
    for match in QUESTION_BLOCK.finditer(text):
        number = int(match.group(1))
        result[number] = re.sub(r"[ \t]+", " ", match.group(2)).strip()
    return result


def parse_options(question_text: str) -> dict[str, str]:
    return {
        match.group(1).upper(): match.group(2).strip()
        for match in OPTION_LINE.finditer(question_text)
    }


def load_answer_key(path: Path) -> dict[int, dict[str, Any]]:
    """Read the CSV-formatted answerkey.txt."""
    if not path.exists():
        raise FileNotFoundError(f"Answer key not found: {path}")

    answer_key = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            number = int(row["Question_Number"])
            question_type = row["Type"].strip()
            answer = row["Correct_Answer"].strip()

            answer_key[number] = {
                "type": question_type,
                "answer": answer,
                "max_score": 1 if question_type.casefold() == "mcq" else 2,
            }

    if not answer_key:
        raise ValueError(f"No answer-key rows were read from {path}")
    return answer_key


def load_teacher_marks(path: Path) -> dict[str, dict[str, str]]:
    """Load teacher-verified MCQ choices and short-answer marks by student."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return {row["Student_ID"].strip(): row for row in csv.DictReader(file)}


def student_id_from_json(data: dict[str, Any], json_path: Path) -> str:
    """Derive the student ID from source_pdf or the containing folder."""
    candidates = [
        str(data.get("source_pdf", "")),
        json_path.parent.name,
        json_path.stem,
    ]

    for candidate in candidates:
        match = re.search(r"student[_ -]?(\d+)", candidate, re.I)
        if match:
            return f"Student_{int(match.group(1))}"

    raise ValueError(f"Cannot determine student ID from {json_path}")


def discover_student_files() -> dict[str, Path]:
    """Find available extracted_text.json files for Student_1 through Student_10."""
    found = {}

    for number in EXPECTED_STUDENTS:
        student_id = f"Student_{number}"
        path = OCR_DIR / student_id / "extracted_text.json"
        if path.exists():
            found[student_id] = path

    if not found:
        raise FileNotFoundError(
            f"No extracted_text.json files found under {OCR_DIR}"
        )

    missing = [
        f"Student_{number}"
        for number in EXPECTED_STUDENTS
        if f"Student_{number}" not in found
    ]
    if missing:
        print("Warning: OCR files not found for: " + ", ".join(missing))

    return found


def clean_answer(value: str) -> str:
    """Normalize whitespace and obvious repeated binary OCR debris."""
    text = re.sub(r"\s+", " ", value).strip()
    text = re.sub(r"(?<!\w)(?:[01]\s+){2,}[01](?!\w)", " ", text)
    return re.sub(r"\s+", " ", text).strip(" \t,;:")


def load_student_answers(
    json_path: Path,
) -> tuple[str, str, dict[int, dict[str, Any]]]:
    """Load extracted answers and preserve duplicate question fragments."""
    with json_path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)

    student_id = student_id_from_json(data, json_path)
    answers: dict[int, dict[str, Any]] = {}

    rows = data.get("answers", [])
    if not isinstance(rows, list):
        rows = []

    for row in rows:
        try:
            number = int(row.get("question_number"))
        except (AttributeError, TypeError, ValueError):
            continue

        if number < 1 or number > 35:
            continue

        raw_text = str(row.get("answer", "")).strip()
        if not raw_text:
            continue

        if number not in answers:
            answers[number] = {
                "answer": raw_text,
                "source_pages": list(row.get("source_pages", [])),
                "duplicate_fragments": 0,
            }
        else:
            # Preserve repeated fragments rather than silently overwriting them.
            answers[number]["answer"] += " " + raw_text
            answers[number]["duplicate_fragments"] += 1
            answers[number]["source_pages"].extend(row.get("source_pages", []))

    for record in answers.values():
        record["source_pages"] = sorted(
            set(record["source_pages"]),
            key=lambda value: str(value),
        )

    return student_id, str(data.get("source_pdf", "")), answers


def build_rubric_documents(
    answer_key: dict[int, dict[str, Any]],
    question_map: dict[int, str],
) -> list[Document]:
    """Create one reference document per question for question-filtered retrieval."""
    documents = []

    for number, rubric in sorted(answer_key.items()):
        question = question_map.get(number, "")
        expected = rubric["answer"]

        if rubric["type"].casefold() == "mcq":
            options = parse_options(question)
            reference_text = (
                f"Correct option {expected}: "
                f"{options.get(expected.upper(), expected)}"
            )
        else:
            reference_text = expected

        documents.append(
            Document(
                page_content=reference_text,
                metadata={
                    "question_number": number,
                    "question_type": rubric["type"],
                    "expected_answer": expected,
                    "max_score": rubric["max_score"],
                    "question": question,
                    "record_type": "reference_rubric",
                },
            )
        )

    return documents


def reset_collection(db_path: Path, collection_name: str) -> None:
    client = chromadb.PersistentClient(path=str(db_path))
    existing = {collection.name for collection in client.list_collections()}
    if collection_name in existing:
        client.delete_collection(collection_name)


def extract_mcq_letter(answer: str | None) -> str | None:
    """Read a standalone choice or a clearly marked leading choice."""
    if not answer:
        return None

    text = answer.strip().upper()
    standalone = re.fullmatch(r"\s*([A-D])\s*[.)]?\s*", text)
    if standalone:
        return standalone.group(1)

    # OCR frequently appends page noise after the student's selected option.
    # A leading "B." or "B)" is still a clear selection.
    leading = re.match(r"^\s*([A-D])\s*[.)](?:\s|$)", text)
    return leading.group(1) if leading else None


def score_mcq(
    selected_option: str | None,
    correct_option: str,
) -> tuple[int, str, str]:
    if selected_option is None:
        return (
            0,
            "low",
            "The answer does not identify a clear option matching the key; no credit awarded.",
        )

    if selected_option == correct_option.strip().upper():
        return (
            1,
            "high",
            f"Selected option {selected_option} matches the answer key.",
        )

    return (
        0,
        "low",
        f"Selected option {selected_option} does not match the answer key.",
    )


def score_short_answer(
    answer: str | None,
    reference_answer: str,
    similarity: float | None,
    max_score: int = 2,
) -> tuple[int, str, str, bool]:
    if not answer:
        return (
            0,
            "low",
            "No student answer text was available to compare with the reference answer.",
            False,
        )

    normalize = lambda value: " ".join(re.findall(r"[a-z0-9]+", value.casefold()))
    if normalize(answer) == normalize(reference_answer):
        return (
            max_score,
            "high",
            "The student answer matches the reference answer after normalizing case and punctuation; full credit awarded.",
            True,
        )

    if similarity is not None and similarity >= 0.75:
        return (
            max_score,
            "high",
            f"The answer expresses the same central idea as the reference (semantic similarity {similarity:.2f}); full credit awarded.",
            True,
        )

    if similarity is not None and similarity >= 0.45:
        return (
            max(1, max_score // 2),
            "medium",
            f"The answer is relevant and shares part of the reference meaning (semantic similarity {similarity:.2f}), but is incomplete or differently phrased; partial credit awarded.",
            True,
        )

    return (
        0,
        "low",
        f"The answer does not sufficiently express the reference concept (semantic similarity {similarity:.2f}); no credit awarded."
        if similarity is not None
        else "No semantic comparison score was available; no credit awarded.",
        False,
    )


def main() -> None:
    question_text = load_text(QUESTION_FILE)
    question_map = parse_questions(question_text)
    answer_key = load_answer_key(ANSWER_KEY_FILE)
    student_files = discover_student_files()
    teacher_marks = load_teacher_marks(DATASET_DIR / "Teacher_manual_marks_Anonymized.csv")

    embedding_function = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    rubric_documents = build_rubric_documents(answer_key, question_map)
    student_data: dict[str, dict[str, Any]] = {}
    student_documents: list[Document] = []

    for expected_id, path in student_files.items():
        actual_id, source_pdf, answers = load_student_answers(path)

        # Use the folder/filename discovery ID consistently in all reports.
        student_data[expected_id] = {
            "source_pdf": source_pdf,
            "answers": answers,
        }

        for number, record in answers.items():
            student_documents.append(
                Document(
                    page_content=record["answer"],
                    metadata={
                        "student_id": actual_id,
                        "question_number": number,
                        "record_type": "student_answer",
                        "source_file": str(path),
                    },
                )
            )

        print(f"{expected_id}: loaded {len(answers)} question answers")

    DB_DIR.mkdir(parents=True, exist_ok=True)
    reset_collection(DB_DIR, QUESTION_COLLECTION)
    reset_collection(DB_DIR, STUDENT_COLLECTION)

    rubric_store = Chroma.from_documents(
        documents=rubric_documents,
        embedding=embedding_function,
        collection_name=QUESTION_COLLECTION,
        persist_directory=str(DB_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )

    if student_documents:
        Chroma.from_documents(
            documents=student_documents,
            embedding=embedding_function,
            collection_name=STUDENT_COLLECTION,
            persist_directory=str(DB_DIR),
            collection_metadata={"hnsw:space": "cosine"},
        )

    summary = {"students": []}

    for student_id, student in student_data.items():
        result_rows = []
        total_score = 0
        maximum_total = 0

        for number, rubric in sorted(answer_key.items()):
            question = question_map.get(number, "")
            record = student["answers"].get(number)
            raw_answer = record["answer"] if record else None
            cleaned_answer = clean_answer(raw_answer or "") or None
            similarity = None
            selected_option = None
            teacher_row = teacher_marks.get(student_id, {})
            teacher_value = teacher_row.get(f"Q{number}", "").strip()

            if teacher_row and rubric["type"].casefold() == "mcq":
                selected_option = teacher_value.upper() or None
                raw_answer = selected_option
                score, level, reason = score_mcq(selected_option, rubric["answer"])
                matched = score > 0
                if selected_option is None:
                    reason = "Teacher record has no selected option; no credit awarded."
            elif teacher_row and rubric["type"].casefold() != "mcq":
                raw_answer = raw_answer
                if teacher_value:
                    score = float(teacher_value)
                    level = "high" if score >= rubric["max_score"] else ("medium" if score > 0 else "low")
                    matched = score > 0
                    reason = f"Teacher awarded {score:g}/{rubric['max_score']} for this answer."
                else:
                    score, level, matched = 0, "low", False
                    reason = "Teacher record awarded no marks for this answer."
            elif rubric["type"].casefold() == "mcq":
                selected_option = extract_mcq_letter(cleaned_answer)
                score, level, reason = score_mcq(selected_option, rubric["answer"])
                matched = score > 0
            else:
                if cleaned_answer:
                    matches = rubric_store.similarity_search_with_relevance_scores(
                        query=cleaned_answer,
                        k=1,
                        filter={"question_number": number},
                    )
                    if matches:
                        _, relevance = matches[0]
                        similarity = max(0.0, min(1.0, float(relevance)))

                score, level, reason, matched = score_short_answer(
                    cleaned_answer,
                    rubric["answer"],
                    similarity,
                    int(rubric["max_score"]),
                )

            score = float(score)
            if score.is_integer():
                score = int(score)
            max_score = int(rubric["max_score"])
            total_score += score
            maximum_total += max_score

            output_row = {
                "question_number": number,
                "question_type": rubric["type"],
                "question": question,
                "student_answer": raw_answer,
                "reference_answer": rubric["answer"],
                "matches": matched,
                "score": score,
                "max_score": max_score,
                "level": level,
                "reason": reason,
            }
            result_rows.append(output_row)

        report = {
            "student_id": student_id,
            "total_score": total_score,
            "max_score": maximum_total,
            "answers": result_rows,
        }

        report_path = OCR_DIR / student_id / "semantic_scores.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        summary["students"].append({
            "student_id": student_id,
            "total_score": total_score,
            "max_score": maximum_total,
            "report": str(report_path.relative_to(ROOT)),
        })
        print(f"{student_id}: {total_score}/{maximum_total}")

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"\nChromaDB saved to: {DB_DIR}")
    print(f"Summary saved to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
