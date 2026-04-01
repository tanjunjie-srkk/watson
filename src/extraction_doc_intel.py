"""
extraction_doc_intel.py
───────────────────────
Uses Azure Document Intelligence (prebuilt-layout) for OCR, converts the result
into the same structured JSON schema that the vision-based OCR agent produces,
then feeds it through the orchestrator for information extraction.

Outputs:
    ocr_output_doc_intel/         – structured OCR result from Document Intelligence
    extraction_output_doc_intel/  – final extraction from the orchestrator

Usage:
    # Single PDF
    python extraction_doc_intel.py --input docs/Inv_1.pdf

    # All PDFs in a directory
    python extraction_doc_intel.py --input docs/

    # Force document type (skip classifier)
    python extraction_doc_intel.py --input docs/LL_1.pdf --type rental

    # Custom output path
    python extraction_doc_intel.py --input docs/Inv_1.pdf --output my_result.json

Environment variables required:
    AZURE_DOC_INTEL_ENDPOINT  – Document Intelligence endpoint URL
    AZURE_DOC_INTEL_KEY       – Document Intelligence API key
    AZURE_OPENAI_ENDPOINT     – (for orchestrator / LLM extraction)
    AZURE_OPENAI_API_KEY      – (for orchestrator / LLM extraction)
"""

import argparse
import json
import os
import sys
from pathlib import Path

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, DocumentAnalysisFeature
from azure.core.credentials import AzureKeyCredential


# ── Config helpers ───────────────────────────────────────────────────────────

def _get_config_value(name: str) -> str | None:
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st
        secret_value = st.secrets.get(name)
        if secret_value:
            return str(secret_value)
    except Exception:
        pass
    return None


def _get_required_env(name: str) -> str:
    value = _get_config_value(name)
    if not value:
        raise RuntimeError(
            f"Missing required config value: {name}. "
            "Set it as an environment variable or Streamlit secret."
        )
    return value


# ── Document Intelligence client ─────────────────────────────────────────────

def _get_di_client() -> DocumentIntelligenceClient:
    endpoint = _get_required_env("AZURE_DOC_INTEL_ENDPOINT")
    key = _get_required_env("AZURE_DOC_INTEL_KEY")
    return DocumentIntelligenceClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key),
    )


# ── Analyze PDF with Document Intelligence ───────────────────────────────────

def _analyze_single_page_pdf(client, pdf_bytes: bytes) -> object:
    """Analyze a single-page PDF with Document Intelligence."""
    poller = client.begin_analyze_document(
        model_id="prebuilt-layout",
        body=pdf_bytes,
        content_type="application/pdf",
        features=[DocumentAnalysisFeature.KEY_VALUE_PAIRS],
    )
    return poller.result()


def analyze_pdf(pdf_path: Path) -> list[tuple[int, object]]:
    """
    Send a PDF to Azure Document Intelligence (prebuilt-layout), processing
    each page individually to guarantee no pages are dropped.

    Returns a list of (page_number, AnalyzeResult) tuples.
    """
    import fitz

    client = _get_di_client()
    doc = fitz.open(str(pdf_path))
    page_count = doc.page_count

    results: list[tuple[int, object]] = []

    if page_count == 1:
        # Single page — send the original PDF directly
        doc.close()
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        result = _analyze_single_page_pdf(client, pdf_bytes)
        results.append((1, result))
    else:
        # Multi-page — split into individual pages and send each separately
        # to prevent DI from dropping pages
        for page_idx in range(page_count):
            single_doc = fitz.open()  # new empty PDF
            single_doc.insert_pdf(doc, from_page=page_idx, to_page=page_idx)
            pdf_bytes = single_doc.tobytes()
            single_doc.close()

            print(f"    Analyzing page {page_idx + 1}/{page_count}...")
            result = _analyze_single_page_pdf(client, pdf_bytes)
            results.append((page_idx + 1, result))

        doc.close()

    return results


# ── Convert DI result → OCR-agent-compatible JSON ────────────────────────────

def _build_table_sections(table) -> list[dict]:
    """Convert a DI table into table_header + table_row sections."""
    sections = []
    if not table.cells:
        return sections

    row_count = table.row_count or 0
    col_count = table.column_count or 0

    # Build a 2D grid
    grid: list[list[str]] = [[""] * col_count for _ in range(row_count)]
    header_rows: set[int] = set()

    for cell in table.cells:
        r = cell.row_index
        c = cell.column_index
        if 0 <= r < row_count and 0 <= c < col_count:
            grid[r][c] = (cell.content or "").strip()
        if getattr(cell, "kind", None) == "columnHeader":
            header_rows.add(r)

    for r_idx, row in enumerate(grid):
        row_text = " | ".join(row)
        if r_idx in header_rows:
            sections.append({"type": "table_header", "content": row_text, "confidence": 0.95})
        else:
            sections.append({"type": "table_row", "content": row_text, "confidence": 0.95})

    return sections


def _build_kv_sections(kv_pairs, page_number: int) -> list[dict]:
    """Convert DI key-value pairs for a given page into key_value sections."""
    sections = []
    kv_lines = []

    for kv in kv_pairs:
        # Check if this KV pair belongs to the target page
        kv_pages = set()
        if kv.key and hasattr(kv.key, "bounding_regions") and kv.key.bounding_regions:
            for br in kv.key.bounding_regions:
                kv_pages.add(br.page_number)
        if kv.value and hasattr(kv.value, "bounding_regions") and kv.value.bounding_regions:
            for br in kv.value.bounding_regions:
                kv_pages.add(br.page_number)

        if page_number not in kv_pages and kv_pages:
            continue

        key_text = (kv.key.content if kv.key else "").strip()
        val_text = (kv.value.content if kv.value else "").strip()
        conf = getattr(kv, "confidence", 0.95) or 0.95

        if key_text or val_text:
            kv_lines.append(f"{key_text} : {val_text}")

    if kv_lines:
        sections.append({
            "type": "key_value",
            "content": "\n".join(kv_lines),
            "confidence": 0.95,
        })

    return sections


def _convert_single_result(result, page_num: int, pdf_path: Path) -> dict:
    """
    Convert a single-page DI AnalyzeResult into one page entry
    for the OCR-agent-compatible JSON schema.
    """
    all_kv_pairs = getattr(result, "key_value_pairs", None) or []
    all_tables = getattr(result, "tables", None) or []
    sections = []

    # ── Key-value pairs ──
    # For single-page results, page_number in bounding_regions is always 1
    kv_sections = _build_kv_sections(all_kv_pairs, 1)
    sections.extend(kv_sections)

    # ── Tables ──
    for table in all_tables:
        sections.extend(_build_table_sections(table))

    # ── Lines (paragraphs) ──
    for page in (result.pages or []):
        lines = []
        for line in (page.lines or []):
            lines.append((line.content or "").strip())
        if lines:
            sections.append({
                "type": "paragraph",
                "content": "\n".join(lines),
                "confidence": 0.98,
            })

    if not sections:
        sections.append({
            "type": "empty",
            "content": "No extractable content on this page.",
            "confidence": 0.0,
        })

    return {
        "page_number": page_num,
        "file_name": f"{pdf_path.stem}_page_{page_num}",
        "sections": sections,
    }


def di_results_to_ocr_json(per_page_results: list[tuple[int, object]], pdf_path: Path) -> dict:
    """
    Convert a list of per-page Azure Document Intelligence AnalyzeResults into
    the same JSON schema the vision-based OCR agent produces, so the orchestrator
    can consume it unchanged.
    """
    pages_out = []
    all_langs: set[str] = set()

    for page_num, result in per_page_results:
        pages_out.append(_convert_single_result(result, page_num, pdf_path))
        langs = _detect_languages(result)
        all_langs.update(langs)

    return {
        "source": "azure_document_intelligence",
        "file": pdf_path.name,
        "pages": pages_out,
        "metadata": {
            "total_pages": len(pages_out),
            "languages_detected": list(all_langs) if all_langs else ["en"],
            "extraction_method": "azure_doc_intel_prebuilt_layout",
        },
    }


def _detect_languages(result) -> list[str]:
    """Extract detected languages from the DI result."""
    langs = set()
    if hasattr(result, "languages") and result.languages:
        for lang in result.languages:
            if hasattr(lang, "locale") and lang.locale:
                langs.add(lang.locale)
    return list(langs) if langs else ["en"]


def has_extractable_content(ocr_dict: dict) -> bool:
    """Return True if at least one page has real content."""
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
) -> Path | None:
    """
    1. Analyze the PDF with Azure Document Intelligence
    2. Convert to orchestrator-compatible OCR JSON
    3. Feed through orchestrator (classify → extract)
    4. Save both OCR and extraction outputs

    Returns the extraction output file path.
    """
    from orchestrator import run as orchestrator_run

    print(f"\n{'═' * 60}")
    print(f"  PDF  : {pdf_path.name}")

    # Step 1 — Document Intelligence OCR (per-page to avoid dropped pages)
    print("  Analyzing with Document Intelligence (prebuilt-layout)...")
    per_page_results = analyze_pdf(pdf_path)

    # Step 2 — Convert to orchestrator-compatible JSON
    ocr_dict = di_results_to_ocr_json(per_page_results, pdf_path)

    if not has_extractable_content(ocr_dict):
        print("  ⚠  No extractable content found — skipping.")
        return None

    total_chars = sum(
        len(s.get("content", ""))
        for p in ocr_dict["pages"]
        for s in p["sections"]
    )
    print(f"  Text : {total_chars:,} chars across {ocr_dict['metadata']['total_pages']} page(s)")

    ocr_json_str = json.dumps(ocr_dict, ensure_ascii=False)

    # Determine output directories
    base_dir = Path(__file__).resolve().parent
    if output_path is None:
        extraction_output_dir = base_dir / "extraction_output_doc_intel"
    else:
        extraction_output_dir = output_path.parent
    extraction_output_dir.mkdir(parents=True, exist_ok=True)

    # Save OCR output
    ocr_output_dir = base_dir / "ocr_output_doc_intel"
    ocr_output_dir.mkdir(parents=True, exist_ok=True)
    ocr_output_path = ocr_output_dir / f"{pdf_path.stem}.json"
    with open(ocr_output_path, "w", encoding="utf-8") as f:
        json.dump(ocr_dict, f, ensure_ascii=False, indent=2)
    print(f"  OCR  : {ocr_output_path}")

    # Step 3 — Orchestrator (classify → extract)
    doc_type, extracted = orchestrator_run(ocr_json_str, forced_type=forced_type)

    # Step 4 — Save extraction output
    if output_path is None:
        output_path = extraction_output_dir / f"{pdf_path.stem}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(extracted, f, ensure_ascii=False, indent=2)

    print(f"  Type : {doc_type}")
    print(f"  Saved: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract data from PDFs using Azure Document Intelligence + orchestrator"
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
        out = Path(args.output) if args.output else None
        process_single_pdf(input_path, output_path=out, forced_type=args.type)

    elif input_path.is_dir():
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
