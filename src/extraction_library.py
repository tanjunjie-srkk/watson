"""
extraction_library.py
─────────────────────
Uses PyMuPDF to extract text from PDFs, wraps it in the same JSON structure
that the vision-based OCR agent produces, then feeds it through the orchestrator
for structured extraction.

Output is saved to  extraction_output_pymupdf/  so you can compare side-by-side
with the original OCR-based results in  extraction_output/.

Usage:
    # Single PDF
    python extraction_library.py --input docs/Inv_1.pdf

    # All PDFs in a directory
    python extraction_library.py --input docs/

    # Force document type (skip classifier)
    python extraction_library.py --input docs/LL_1.pdf --type rental

    # Custom output path
    python extraction_library.py --input docs/Inv_1.pdf --output my_result.json
"""

import argparse
import json
import sys
from pathlib import Path

import fitz  # PyMuPDF


# ── PyMuPDF text extraction ─────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: Path) -> dict:
    """
    Extract text from a PDF using PyMuPDF and wrap it in the same JSON schema
    that the vision-based OCR agent produces so the orchestrator / classifier
    can consume it without changes.

    Returns a dict shaped like:
        {
            "source": "pymupdf",
            "file": "<filename>",
            "pages": [
                {
                    "page_number": 1,
                    "file_name": "<filename>",
                    "sections": [
                        {"type": "paragraph", "content": "...", "confidence": 1.0}
                    ]
                },
                ...
            ],
            "metadata": {
                "total_pages": N,
                "languages_detected": ["en"],
                "extraction_method": "pymupdf_text"
            }
        }
    """
    doc = fitz.open(str(pdf_path))
    pages = []

    for i, page in enumerate(doc):
        text = page.get_text("text")  # plain UTF-8 text

        # Build sections list — one section per page for plain text.
        # If the page has no text (scanned image), flag it.
        if text.strip():
            sections = [
                {
                    "type": "paragraph",
                    "content": text.strip(),
                    "confidence": 1.0,
                }
            ]
        else:
            sections = [
                {
                    "type": "empty",
                    "content": "No extractable text — page may be a scanned image.",
                    "confidence": 0.0,
                }
            ]

        pages.append(
            {
                "page_number": i + 1,
                "file_name": f"{pdf_path.stem}_page_{i + 1}",
                "sections": sections,
            }
        )

    doc.close()

    return {
        "source": "pymupdf",
        "file": pdf_path.name,
        "pages": pages,
        "metadata": {
            "total_pages": len(pages),
            "languages_detected": ["en"],
            "extraction_method": "pymupdf_text",
        },
    }


def has_extractable_text(ocr_dict: dict) -> bool:
    """Return True if at least one page has real text content."""
    for page in ocr_dict.get("pages", []):
        for section in page.get("sections", []):
            if section.get("type") != "empty" and section.get("content", "").strip():
                return True
    return False


# ── Main logic ───────────────────────────────────────────────────────────────

def process_single_pdf(
    pdf_path: Path,
    output_path: Path | None = None,
    forced_type: str | None = None,
) -> Path:
    """
    1. Extract text with PyMuPDF
    2. Feed the resulting JSON string into orchestrator.run()
    3. Save the extracted result

    Returns the output file path.
    """
    # Lazy import so the module-level Azure client is only created when needed
    from orchestrator import run as orchestrator_run

    print(f"\n{'═' * 60}")
    print(f"  PDF  : {pdf_path.name}")

    # Step 1 — PyMuPDF text extraction
    ocr_dict = extract_text_from_pdf(pdf_path)

    if not has_extractable_text(ocr_dict):
        print("  ⚠  No extractable text found (scanned PDF?) — skipping.")
        return None

    total_chars = sum(
        len(s.get("content", ""))
        for p in ocr_dict["pages"]
        for s in p["sections"]
    )
    print(f"  Text : {total_chars:,} chars across {ocr_dict['metadata']['total_pages']} page(s)")

    ocr_json_str = json.dumps(ocr_dict, ensure_ascii=False)

    # Determine output directory
    if output_path is None:
        output_dir = Path(__file__).resolve().parent / "extraction_output_pymupdf"
    else:
        output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save OCR output (PyMuPDF text extraction)
    ocr_output_dir = Path(__file__).resolve().parent / "ocr_output_pymupdf"
    ocr_output_dir.mkdir(parents=True, exist_ok=True)
    ocr_output_path = ocr_output_dir / f"{pdf_path.stem}.json"
    with open(ocr_output_path, "w", encoding="utf-8") as f:
        json.dump(ocr_dict, f, ensure_ascii=False, indent=2)
    print(f"  OCR  : {ocr_output_path}")

    # Step 2 — Orchestrator (classify → extract)
    doc_type, extracted = orchestrator_run(ocr_json_str, forced_type=forced_type)

    # Step 3 — Save extraction output
    if output_path is None:
        output_path = output_dir / f"{pdf_path.stem}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(extracted, f, ensure_ascii=False, indent=2)

    print(f"  Type : {doc_type}")
    print(f"  Saved: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract data from PDFs using PyMuPDF text + orchestrator"
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to a single PDF or a directory of PDFs",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output file path (only for single-PDF mode)",
    )
    parser.add_argument(
        "--type", "-t", default=None,
        help="Force document type, skip classifier (e.g. rental, utility, commercial_invoice)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_file() and input_path.suffix.lower() == ".pdf":
        # ── Single PDF ──
        out = args.output and Path(args.output)
        process_single_pdf(input_path, output_path=out, forced_type=args.type)

    elif input_path.is_dir():
        # ── All PDFs in directory ──
        pdfs = sorted(input_path.glob("*.pdf"))
        if not pdfs:
            print(f"No PDF files found in: {input_path}", file=sys.stderr)
            sys.exit(1)

        print(f"Found {len(pdfs)} PDF(s) in {input_path}\n")
        for pdf in pdfs:
            try:
                process_single_pdf(pdf, forced_type=args.type)
            except Exception as e:
                print(f"  ERROR processing {pdf.name}: {e}", file=sys.stderr)

    else:
        print(f"ERROR: {input_path} is not a PDF file or directory.", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'═' * 60}")
    print("Done.")


if __name__ == "__main__":
    main()