"""
hybrid_ocr.py
─────────────
Hybrid cross-validation pipeline that merges:
  • Azure Document Intelligence  → accurate key-value / numeric fields
  • GPT Vision OCR               → accurate table structure (nested rows, barcodes)

The merge strategy:
  1. Run Doc Intel on the original PDF → get digit-perfect key-value readings
  2. Run GPT Vision on the rendered images → get correct table layout
  3. Use GPT Vision output as the BASE (preserving table rows)
  4. REPLACE key-value fields in the vision output with Doc Intel's more accurate values
  5. Feed the merged OCR JSON to the orchestrator for extraction

Usage:
    # Full pipeline: PDF → hybrid OCR → extract
    python hybrid_ocr.py --input docs/Inv_1.pdf

    # Force document type (skip classifier)
    python hybrid_ocr.py --input docs/Inv_1.pdf --type commercial_invoice

    # Merge from pre-existing OCR outputs (skip re-running both pipelines)
    python hybrid_ocr.py --vision-ocr ocr_output/Inv_1.json --di-ocr ocr_output_doc_intel/Inv_1.json

    # All PDFs in a directory
    python hybrid_ocr.py --input docs/
"""

import argparse
import copy
import json
import re
import sys
from pathlib import Path


# ── Key normalization ─────────────────────────────────────────────────────────

def _normalize_key(key: str) -> str:
    """Normalize a key for comparison.

    Handles variations like:
      "Bill Of Lading:"  vs  "Bill Of Lading"
      "BU PO Number"     vs  "BU PO Number :"
      Extra whitespace, trailing colons, mixed case
    """
    k = key.strip()
    # Remove trailing colons and whitespace
    k = k.rstrip(":").strip()
    # Collapse multiple spaces
    k = re.sub(r"\s+", " ", k)
    return k.lower()


# ── Extract KV pairs from Doc Intel output ───────────────────────────────────

def _parse_kv_line(line: str) -> tuple[str, str] | None:
    """Parse a single 'Key : Value' line, handling double colons from Doc Intel.

    Handles:
      "BU PO Number : 18082166"
      "BU PO Number : : 18082166"
      "Bill Of Lading: : CNSHA0001548220"
      "CI Number : : 26001099"
    """
    # Try splitting on " : : " first (Doc Intel double colon pattern)
    if " : : " in line:
        key, _, val = line.partition(" : : ")
        return key.strip(), val.strip()

    # Try splitting on " : " (standard pattern)
    if " : " in line:
        key, _, val = line.partition(" : ")
        # Handle case where val starts with ": " (another double colon variant)
        val = val.lstrip(": ").strip()
        return key.strip(), val.strip()

    return None


def extract_di_kv_per_page(di_ocr: dict) -> dict[int, dict[str, str]]:
    """Extract key-value pairs from Doc Intel OCR, organized by page.

    Returns:
        dict mapping page_number → {normalized_key: value}
    """
    per_page_kv: dict[int, dict[str, str]] = {}

    for page in di_ocr.get("pages", []):
        page_num = page.get("page_number", 0)
        kv_dict: dict[str, str] = {}

        for section in page.get("sections", []):
            # Extract from key_value sections (primary source — highest accuracy)
            if section.get("type") == "key_value":
                for line in section["content"].split("\n"):
                    parsed = _parse_kv_line(line)
                    if parsed:
                        key, val = parsed
                        if val:  # Only include if value is non-empty
                            norm_key = _normalize_key(key)
                            kv_dict[norm_key] = val

            # Also extract from paragraph sections (backup — paragraphs contain
            # the full page text including KV-like lines)
            elif section.get("type") == "paragraph":
                for line in section["content"].split("\n"):
                    parsed = _parse_kv_line(line)
                    if parsed:
                        key, val = parsed
                        if val and _normalize_key(key) not in kv_dict:
                            kv_dict[_normalize_key(key)] = val

        per_page_kv[page_num] = kv_dict

    return per_page_kv


def build_di_consensus(per_page_kv: dict[int, dict[str, str]]) -> dict[str, str]:
    """Build a consensus KV dict across all pages (Doc Intel reads the same
    header fields on every page, so we have multiple readings of each field).

    Uses majority voting when pages disagree.
    """
    from collections import Counter

    # Collect all values for each key across pages
    all_key_values: dict[str, list[str]] = {}
    for page_num, kv in per_page_kv.items():
        for norm_key, val in kv.items():
            all_key_values.setdefault(norm_key, []).append(val)

    # Majority vote
    consensus: dict[str, str] = {}
    for key, values in all_key_values.items():
        counts = Counter(values)
        winner, count = counts.most_common(1)[0]
        consensus[key] = winner

    return consensus


# ── Flatten GPT Vision output to standard pages format ───────────────────────

def flatten_vision_output(vision_ocr: dict) -> dict:
    """Convert GPT Vision's wrapped output format to the flat pages[] format
    that the orchestrator expects.

    GPT Vision format:
        {"mode": "...", "results": [{"page_number": N, "file": "...", "model_output": {"pages": [...]}}]}

    Standard format:
        {"source": "...", "file": "...", "pages": [...], "metadata": {...}}
    """
    # If already in flat format, return as-is
    if "pages" in vision_ocr and "results" not in vision_ocr:
        return copy.deepcopy(vision_ocr)

    pages = []
    results = vision_ocr.get("results", [])

    for result in results:
        model_output = result.get("model_output", {})
        # Handle case where model_output is a string (parse error)
        if isinstance(model_output, str):
            try:
                model_output = json.loads(model_output)
            except Exception:
                continue

        for page in model_output.get("pages", []):
            # Fix page_number to be sequential across the full document
            page_copy = copy.deepcopy(page)
            page_copy["page_number"] = result.get("page_number", page.get("page_number", 1))
            page_copy["file_name"] = result.get("file", page.get("file_name", ""))
            pages.append(page_copy)

    return {
        "source": "hybrid_gpt_vision_base",
        "file": vision_ocr.get("file", ""),
        "pages": pages,
        "metadata": {
            "total_pages": len(pages),
            "languages_detected": ["en"],
            "extraction_method": "hybrid_di_vision",
        },
    }


# ── Merge: replace KV fields in vision output with Doc Intel values ──────────

def _should_replace(old_val: str, new_val: str) -> bool:
    """Decide whether a Doc Intel value should replace a GPT Vision value.

    Guards against replacing good vision data with broken DI data:
    - Skip if old_val has digits but new_val lost them (e.g. "USD 30,752.88" → "USD")
    - Skip if new_val is dramatically shorter (< 40% of old_val length)
    - Skip if both are identical
    """
    if old_val == new_val:
        return False  # No change needed

    old_has_digits = bool(re.search(r"\d", old_val))
    new_has_digits = bool(re.search(r"\d", new_val))

    # If old value contains digits and new value doesn't, the DI parsing
    # likely split the value across lines — don't replace
    if old_has_digits and not new_has_digits:
        return False

    # If the replacement is dramatically shorter, it's probably incomplete
    if len(old_val) > 5 and len(new_val) < len(old_val) * 0.4:
        return False

    return True


def _replace_kv_in_text(text: str, consensus_kv: dict[str, str]) -> tuple[str, list[str]]:
    """Replace key-value fields in a text block using Doc Intel consensus values.

    Returns (new_text, list_of_replacements_made).
    """
    replacements = []
    new_lines = []

    for line in text.split("\n"):
        parsed = _parse_kv_line(line)
        if parsed:
            key, old_val = parsed
            norm_key = _normalize_key(key)
            if norm_key in consensus_kv:
                di_val = consensus_kv[norm_key]
                if _should_replace(old_val, di_val):
                    replacements.append(
                        f"  {key}: '{old_val}' → '{di_val}'"
                    )
                    new_lines.append(f"{key} : {di_val}")
                else:
                    new_lines.append(line)  # Keep original
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    return "\n".join(new_lines), replacements


def merge_ocr_outputs(vision_ocr: dict, di_ocr: dict, verbose: bool = True) -> dict:
    """Merge GPT Vision OCR (table structure) with Doc Intel OCR (key-value accuracy).

    Strategy:
      - Use the GPT Vision output as BASE (correct table headers + rows)
      - Replace key_value sections with Doc Intel's accurate readings
      - Also scan paragraph/subtotal sections for embedded KV lines and fix them
      - Leave table_header, table_row, address, header, footer sections untouched

    Args:
        vision_ocr: GPT Vision OCR output dict (may be wrapped or flat format).
        di_ocr: Doc Intel OCR output dict.
        verbose: Print replacement details.

    Returns:
        Merged OCR dict in the standard flat pages format.
    """
    # Step 1: Flatten vision output
    merged = flatten_vision_output(vision_ocr)

    # Step 2: Extract KV pairs from Doc Intel and build cross-page consensus
    di_per_page_kv = extract_di_kv_per_page(di_ocr)
    consensus_kv = build_di_consensus(di_per_page_kv)

    if verbose:
        print(f"\n  ── Doc Intel consensus KV fields ({len(consensus_kv)} keys) ──")
        for k, v in sorted(consensus_kv.items()):
            print(f"    {k}: {v}")

    # Step 3: Replace KV fields in each page of the vision output
    total_replacements = []

    # Section types that may contain KV lines to fix
    KV_SECTION_TYPES = {"key_value", "paragraph", "subtotal"}

    for page in merged.get("pages", []):
        page_num = page.get("page_number", "?")

        for section in page.get("sections", []):
            section_type = section.get("type", "")

            if section_type in KV_SECTION_TYPES:
                new_content, replacements = _replace_kv_in_text(
                    section.get("content", ""), consensus_kv
                )
                if replacements:
                    section["content"] = new_content
                    # Boost confidence since we've cross-validated with Doc Intel
                    section["confidence"] = min(
                        section.get("confidence", 0.95) + 0.05, 1.0
                    )
                    section["cross_validated"] = True
                    for r in replacements:
                        total_replacements.append(f"  Page {page_num} [{section_type}]: {r}")

    if verbose:
        if total_replacements:
            print(f"\n  ── Cross-validation replacements ({len(total_replacements)}) ──")
            for r in total_replacements:
                print(r)
        else:
            print("\n  ── No replacements needed (vision and DI agree) ──")

    # Update metadata
    merged["source"] = "hybrid_di_vision"
    merged["metadata"]["extraction_method"] = "hybrid_di_vision"

    return merged


# ── Full pipeline: PDF → hybrid OCR → orchestrator extraction ────────────────

def process_single_pdf(
    pdf_path: Path,
    output_path: Path | None = None,
    forced_type: str | None = None,
    batch_vision: bool = True,
) -> Path | None:
    """
    Full hybrid pipeline:
      1. Run Doc Intel on the PDF → accurate KV fields
      2. Convert PDF to images + run GPT Vision OCR → accurate table structure
      3. Merge the two OCR outputs
      4. Feed merged OCR JSON to the orchestrator (classify → extract)
      5. Save OCR + extraction outputs

    Returns the extraction output file path.
    """
    from pdf_to_images import pdf_to_images
    from ocr_agent import ocr_images_with_chat_model, _maybe_parse_json
    from extraction_doc_intel import analyze_pdf, di_results_to_ocr_json
    from orchestrator import run as orchestrator_run

    print(f"\n{'═' * 70}")
    print(f"  HYBRID PIPELINE: {pdf_path.name}")
    print(f"{'═' * 70}")

    # ── Step 1: Doc Intel OCR ──────────────────────────────────────────────
    print("\n  [1/4] Running Azure Document Intelligence (prebuilt-layout)...")
    per_page_results = analyze_pdf(pdf_path)
    di_ocr = di_results_to_ocr_json(per_page_results, pdf_path)

    di_chars = sum(
        len(s.get("content", ""))
        for p in di_ocr["pages"]
        for s in p["sections"]
    )
    print(f"         {di_chars:,} chars across {di_ocr['metadata']['total_pages']} page(s)")

    # ── Step 2: GPT Vision OCR ─────────────────────────────────────────────
    print("\n  [2/4] Running GPT Vision OCR on document images...")

    # Convert PDF to images (or reuse existing ones)
    base_dir = Path(__file__).resolve().parent
    images_dir = base_dir / "docs" / f"{pdf_path.stem}_images"
    if images_dir.exists() and list(images_dir.glob("*.png")):
        image_paths = sorted(images_dir.glob("*.png"))
        print(f"         Reusing {len(image_paths)} existing image(s) from {images_dir.name}/")
    else:
        image_paths = pdf_to_images(pdf_path, output_dir=images_dir)
        print(f"         Generated {len(image_paths)} image(s)")

    user_prompt = (
        "Transcribe ALL visible text from this document image exactly as it appears. "
        "Output the result as a single valid JSON object following the schema in your instructions. "
        "Do NOT interpret, summarize, or calculate anything. "
        "For every section, set a realistic confidence score — lower it if any character is uncertain. "
        "Preserve all numbers, punctuation, and formatting exactly."
    )

    if batch_vision:
        # Send all images in one request (preserves cross-page context)
        raw_vision = ocr_images_with_chat_model(image_paths, user_prompt)
        vision_output = _maybe_parse_json(raw_vision)
        if isinstance(vision_output, str):
            print(f"  WARNING: Vision OCR returned non-JSON: {vision_output[:200]}")
            vision_ocr = {"pages": [], "metadata": {}}
        else:
            # Wrap in standard flat format
            vision_ocr = vision_output
    else:
        # Per-image mode (for very large documents)
        from ocr_agent import ocr_image_with_chat_model
        results = []
        for idx, img_path in enumerate(image_paths, 1):
            print(f"         Processing page {idx}/{len(image_paths)}...")
            raw = ocr_image_with_chat_model(img_path, user_prompt)
            results.append({
                "page_number": idx,
                "file": img_path.name,
                "model_output": _maybe_parse_json(raw),
            })
        vision_ocr = {"mode": "per_image", "results": results}

    vision_chars = 0
    if "pages" in vision_ocr:
        for p in vision_ocr["pages"]:
            for s in p.get("sections", []):
                vision_chars += len(s.get("content", ""))
    elif "results" in vision_ocr:
        for r in vision_ocr.get("results", []):
            mo = r.get("model_output", {})
            if isinstance(mo, dict):
                for p in mo.get("pages", []):
                    for s in p.get("sections", []):
                        vision_chars += len(s.get("content", ""))
    print(f"         {vision_chars:,} chars from GPT Vision")

    # ── Step 3: Merge ──────────────────────────────────────────────────────
    print("\n  [3/4] Merging Doc Intel KV accuracy + GPT Vision table structure...")
    merged_ocr = merge_ocr_outputs(vision_ocr, di_ocr, verbose=True)

    # Save all OCR outputs
    ocr_output_dir = base_dir / "ocr_output_hybrid"
    ocr_output_dir.mkdir(parents=True, exist_ok=True)

    # Save Doc Intel OCR
    with open(ocr_output_dir / f"{pdf_path.stem}_di.json", "w", encoding="utf-8") as f:
        json.dump(di_ocr, f, ensure_ascii=False, indent=2)

    # Save Vision OCR
    with open(ocr_output_dir / f"{pdf_path.stem}_vision.json", "w", encoding="utf-8") as f:
        json.dump(vision_ocr, f, ensure_ascii=False, indent=2)

    # Save Merged OCR
    merged_ocr_path = ocr_output_dir / f"{pdf_path.stem}.json"
    with open(merged_ocr_path, "w", encoding="utf-8") as f:
        json.dump(merged_ocr, f, ensure_ascii=False, indent=2)
    print(f"         Saved merged OCR → {merged_ocr_path}")

    # ── Step 4: Orchestrator (classify → extract) ─────────────────────────
    print("\n  [4/4] Running orchestrator (classify → extract)...")
    ocr_json_str = json.dumps(merged_ocr, ensure_ascii=False)
    doc_type, extracted = orchestrator_run(ocr_json_str, forced_type=forced_type)

    # Save extraction output
    extraction_output_dir = base_dir / "extraction_output_hybrid"
    extraction_output_dir.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        output_path = extraction_output_dir / f"{pdf_path.stem}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(extracted, f, ensure_ascii=False, indent=2)

    print(f"\n  Type : {doc_type}")
    print(f"  Saved: {output_path}")
    print(f"{'═' * 70}")

    return output_path


def merge_from_files(
    vision_ocr_path: Path,
    di_ocr_path: Path,
    output_path: Path | None = None,
    forced_type: str | None = None,
) -> Path | None:
    """Merge from pre-existing OCR output files (skip re-running the pipelines).

    Useful when you've already run both pipelines separately and just want
    to merge + re-extract.
    """
    from orchestrator import run as orchestrator_run

    print(f"\n{'═' * 70}")
    print(f"  HYBRID MERGE (from existing OCR files)")
    print(f"{'═' * 70}")

    # Load both OCR outputs
    with open(vision_ocr_path, "r", encoding="utf-8") as f:
        vision_ocr = json.load(f)
    print(f"  Vision OCR: {vision_ocr_path}")

    with open(di_ocr_path, "r", encoding="utf-8") as f:
        di_ocr = json.load(f)
    print(f"  Doc Intel:  {di_ocr_path}")

    # Merge
    merged_ocr = merge_ocr_outputs(vision_ocr, di_ocr, verbose=True)

    # Save merged OCR
    base_dir = Path(__file__).resolve().parent
    ocr_output_dir = base_dir / "ocr_output_hybrid"
    ocr_output_dir.mkdir(parents=True, exist_ok=True)

    stem = vision_ocr_path.stem.replace("_vision", "")
    merged_ocr_path = ocr_output_dir / f"{stem}.json"
    with open(merged_ocr_path, "w", encoding="utf-8") as f:
        json.dump(merged_ocr, f, ensure_ascii=False, indent=2)
    print(f"\n  Merged OCR → {merged_ocr_path}")

    # Orchestrator
    print("\n  Running orchestrator (classify → extract)...")
    ocr_json_str = json.dumps(merged_ocr, ensure_ascii=False)
    doc_type, extracted = orchestrator_run(ocr_json_str, forced_type=forced_type)

    # Save extraction
    extraction_output_dir = base_dir / "extraction_output_hybrid"
    extraction_output_dir.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        output_path = extraction_output_dir / f"{stem}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(extracted, f, ensure_ascii=False, indent=2)

    print(f"\n  Type : {doc_type}")
    print(f"  Saved: {output_path}")
    print(f"{'═' * 70}")

    return output_path


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hybrid OCR: merge Doc Intel KV accuracy + GPT Vision table structure"
    )

    # Mode 1: Full pipeline from PDF
    parser.add_argument(
        "--input", default=None,
        help="Path to a single PDF or directory of PDFs (runs full pipeline)",
    )

    # Mode 2: Merge from existing OCR files
    parser.add_argument(
        "--vision-ocr", default=None,
        help="Path to existing GPT Vision OCR output JSON",
    )
    parser.add_argument(
        "--di-ocr", default=None,
        help="Path to existing Doc Intel OCR output JSON",
    )

    # Common options
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output file path (only for single-file mode)",
    )
    parser.add_argument(
        "--type", "-t", default=None,
        help="Force document type, skip classifier (e.g. commercial_invoice, rental, utility)",
    )
    parser.add_argument(
        "--per-image", action="store_true",
        help="Send images one-by-one to vision model instead of batch (for large documents)",
    )

    args = parser.parse_args()

    # Mode 2: Merge from existing files
    if args.vision_ocr and args.di_ocr:
        merge_from_files(
            vision_ocr_path=Path(args.vision_ocr),
            di_ocr_path=Path(args.di_ocr),
            output_path=Path(args.output) if args.output else None,
            forced_type=args.type,
        )
        return

    # Mode 1: Full pipeline from PDF
    if not args.input:
        parser.error("Provide either --input (PDF path) or both --vision-ocr and --di-ocr")

    input_path = Path(args.input)

    if input_path.is_file() and input_path.suffix.lower() == ".pdf":
        out = Path(args.output) if args.output else None
        process_single_pdf(
            input_path,
            output_path=out,
            forced_type=args.type,
            batch_vision=not args.per_image,
        )

    elif input_path.is_dir():
        pdfs = sorted(input_path.glob("*.pdf"))
        if not pdfs:
            print(f"No PDF files found in: {input_path}", file=sys.stderr)
            sys.exit(1)

        print(f"Found {len(pdfs)} PDF(s) in {input_path}\n")
        for pdf in pdfs:
            try:
                process_single_pdf(
                    pdf,
                    forced_type=args.type,
                    batch_vision=not args.per_image,
                )
            except Exception as e:
                print(f"  ERROR processing {pdf.name}: {e}", file=sys.stderr)

    else:
        print(f"ERROR: {input_path} is not a PDF file or directory.", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'═' * 70}")
    print("Done.")


if __name__ == "__main__":
    main()
