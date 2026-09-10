"""Ingestion module — load and extract raw text from resume files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from src.config import PDF_EXTENSIONS, SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)


def extract_pdf_text(file_path: Path) -> Optional[str]:
    """Extract text from a PDF file using PyMuPDF (fitz)."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(file_path))
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        return "\n".join(text_parts).strip()
    except Exception as e:
        logger.warning(f"Failed to extract PDF {file_path.name}: {e}")
        return None


def extract_docx_text(file_path: Path) -> Optional[str]:
    """Extract text from a DOCX file."""
    try:
        import docx
        doc = docx.Document(str(file_path))
        return "\n".join(p.text for p in doc.paragraphs).strip()
    except Exception as e:
        logger.warning(f"Failed to extract DOCX {file_path.name}: {e}")
        return None


def extract_txt_text(file_path: Path) -> Optional[str]:
    """Extract text from a plain text file."""
    try:
        return file_path.read_text(encoding="utf-8").strip()
    except Exception as e:
        logger.warning(f"Failed to extract TXT {file_path.name}: {e}")
        return None


EXTRACTORS = {
    ".pdf": extract_pdf_text,
    ".docx": extract_docx_text,
    ".txt": extract_txt_text,
}


def extract_text_from_file(file_path: Path) -> Optional[str]:
    """Route to the appropriate extractor based on file extension."""
    ext = file_path.suffix.lower()
    if ext not in EXTRACTORS:
        logger.warning(f"Unsupported file type: {ext} for {file_path.name}")
        return None
    return EXTRACTORS[ext](file_path)


def load_resumes(input_dir: Path) -> list[dict]:
    """
    Scan input_dir for resume files, extract text from each.

    Returns a list of dicts: [{"filename": str, "raw_text": str, "path": Path}]
    """
    results = []
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return results

    files = sorted(input_dir.iterdir())
    seen_names: set[str] = set()

    for file_path in files:
        if not file_path.is_file():
            continue
        ext = file_path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue

        # Handle duplicate filenames
        name = file_path.name
        if name in seen_names:
            stem = file_path.stem
            counter = 1
            while f"{stem}_{counter}{ext}" in seen_names:
                counter += 1
            name = f"{stem}_{counter}{ext}"
        seen_names.add(name)

        raw_text = extract_text_from_file(file_path)
        if raw_text:
            results.append({
                "filename": name,
                "raw_text": raw_text,
                "path": file_path,
            })
            logger.info(f"Loaded: {name} ({len(raw_text)} chars)")
        else:
            results.append({
                "filename": name,
                "raw_text": "",
                "path": file_path,
            })
            logger.warning(f"Failed to extract text from: {name}")

    return results
