"""
Orchestrator: classifies an OCR JSON file and routes to the correct extraction agent.

Usage:
    python orchestrator.py <ocr_json_file>
    python orchestrator.py <ocr_json_file> --output result.json
    python orchestrator.py <ocr_json_file> --type commercial_invoice   # skip classification
"""

import argparse
import json
import sys
from pathlib import Path

from agents import call_extraction_agent, maybe_parse_json
from agents.classifier import classify_document
from agents import extraction_invoice
from agents import extraction_travel
from agents import extraction_rental
from agents import extraction_hotel
from agents import extraction_utility
from agents import extraction_soa
from agents import extraction_bank

# Registry: maps classifier label → (SYSTEM_PROMPT, USER_PROMPT)
AGENT_REGISTRY: dict[str, tuple[str, str]] = {
    "commercial_invoice": (extraction_invoice.SYSTEM_PROMPT, extraction_invoice.USER_PROMPT),
    "credit_note":        (extraction_invoice.SYSTEM_PROMPT, extraction_invoice.USER_PROMPT),  # reuse invoice schema
    "travel":             (extraction_travel.SYSTEM_PROMPT,  extraction_travel.USER_PROMPT),
    "rental":             (extraction_rental.SYSTEM_PROMPT,  extraction_rental.USER_PROMPT),
    "hotel":              (extraction_hotel.SYSTEM_PROMPT,   extraction_hotel.USER_PROMPT),
    "utility":            (extraction_utility.SYSTEM_PROMPT, extraction_utility.USER_PROMPT),
    "soa":                (extraction_soa.SYSTEM_PROMPT,     extraction_soa.USER_PROMPT),
    "bank_statement":     (extraction_bank.SYSTEM_PROMPT,    extraction_bank.USER_PROMPT),
}

# Fallback: generic extraction (uses the current extraction_agent.py style)
FALLBACK_SYSTEM_PROMPT = extraction_invoice.SYSTEM_PROMPT
FALLBACK_USER_PROMPT = extraction_invoice.USER_PROMPT


def run(ocr_json_str: str, forced_type: str | None = None) -> tuple[str, object]:
    """
    Classify and extract.

    Args:
        ocr_json_str: Raw OCR JSON string.
        forced_type: If set, skip classification and use this type directly.

    Returns:
        (document_type, extracted_data)
    """
    # 1. Classify
    if forced_type:
        doc_type = forced_type
        print(f"  Document type (forced): {doc_type}")
    else:
        doc_type = classify_document(ocr_json_str)
        print(f"  Document type (classified): {doc_type}")

    # 2. Route to agent
    if doc_type in AGENT_REGISTRY:
        system_prompt, user_prompt = AGENT_REGISTRY[doc_type]
    else:
        print(f"  WARNING: Unknown type '{doc_type}', using fallback (commercial_invoice) agent.")
        system_prompt, user_prompt = FALLBACK_SYSTEM_PROMPT, FALLBACK_USER_PROMPT

    # 3. Extract
    raw_result = call_extraction_agent(system_prompt, user_prompt, ocr_json_str)
    parsed = maybe_parse_json(raw_result)

    # Inject classification into result if it's a dict
    if isinstance(parsed, dict) and "document_type" not in parsed:
        parsed["document_type"] = doc_type

    return doc_type, parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Orchestrator: classify + extract from OCR JSON")
    parser.add_argument("--input", help="Path to OCR output JSON file")
    parser.add_argument("--output", "-o", default=None, help="Output file path")
    parser.add_argument("--type", "-t", default=None,
                        choices=list(AGENT_REGISTRY.keys()) + ["unknown"],
                        help="Force document type (skip classification)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    ocr_json_str = input_path.read_text(encoding="utf-8")
    print(f"Processing: {input_path.name}")

    doc_type, extracted = run(ocr_json_str, forced_type=args.type)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = input_path.parent.parent / "extraction_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{input_path.stem}_extracted.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(extracted, f, ensure_ascii=False, indent=2)

    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()
