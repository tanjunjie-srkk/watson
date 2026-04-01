"""
pdfplumber_demo.py
──────────────────
Quick demo showing what pdfplumber can extract from Inv_1.pdf:
  1. Plain text per page
  2. Words with bounding boxes
  3. Text lines with positions
  4. Table detection and extraction
  5. Basic page geometry (lines, rects)

Usage:
    python pdfplumber_demo.py
    python pdfplumber_demo.py --pdf docs/Inv_1.pdf
"""

import argparse
import json
from pathlib import Path

import pdfplumber


def demo(pdf_path: Path) -> None:
    print("=" * 70)
    print(f"PDF: {pdf_path.name}")
    print("=" * 70)

    with pdfplumber.open(pdf_path) as pdf:
        print(f"Total pages: {len(pdf.pages)}\n")

        for page_num, page in enumerate(pdf.pages, start=1):
            print("─" * 70)
            print(f"PAGE {page_num}  (size: {round(page.width)}w x {round(page.height)}h pts)")
            print("─" * 70)

            # ── 1. Plain text ──────────────────────────────────────────────
            print("\n[1] PLAIN TEXT")
            text = page.extract_text()
            if text:
                # Show first 500 chars to keep output readable
                preview = text[:500] + ("..." if len(text) > 500 else "")
                print(preview)
            else:
                print("  (no embedded text — likely a scanned page)")

            # ── 2. Words with bounding boxes ───────────────────────────────
            print("\n[2] WORDS WITH BOUNDING BOXES (first 10)")
            words = page.extract_words()
            for w in words[:10]:
                print(f"  '{w['text']:30s}'  x0={w['x0']:6.1f}  y0={w['top']:6.1f}  "
                      f"x1={w['x1']:6.1f}  y1={w['bottom']:6.1f}")
            if len(words) > 10:
                print(f"  ... ({len(words)} words total)")

            # ── 3. Text lines with positions ───────────────────────────────
            print("\n[3] TEXT LINES WITH POSITIONS (first 8)")
            lines = page.extract_text_lines()
            for ln in lines[:8]:
                print(f"  y={ln['top']:6.1f}  '{ln['text'][:60]}'")
            if len(lines) > 8:
                print(f"  ... ({len(lines)} lines total)")

            # ── 4. Table detection ─────────────────────────────────────────
            print("\n[4] TABLES")
            tables = page.extract_tables()
            if tables:
                print(f"  Found {len(tables)} table(s)")
                for t_idx, table in enumerate(tables, start=1):
                    print(f"\n  Table {t_idx}: {len(table)} rows x {len(table[0]) if table else 0} cols")
                    for row_idx, row in enumerate(table):
                        # Show max 5 rows per table
                        if row_idx >= 5:
                            print(f"    ... ({len(table) - 5} more rows)")
                            break
                        cells = [str(c or "").strip()[:20] for c in row]
                        print(f"    Row {row_idx:2d}: {cells}")
            else:
                print("  No tables detected on this page")

            # ── 5. Page geometry (lines & rectangles) ──────────────────────
            print("\n[5] GEOMETRY")
            print(f"  Lines:      {len(page.lines)}")
            print(f"  Rectangles: {len(page.rects)}")
            print(f"  Curves:     {len(page.curves)}")
            print(f"  Images:     {len(page.images)}")

            # Show a few rectangles (potential table borders or boxes)
            if page.rects:
                print("  First 3 rectangles:")
                for r in page.rects[:3]:
                    print(f"    x0={r['x0']:.1f} y0={r['top']:.1f} "
                          f"x1={r['x1']:.1f} y1={r['bottom']:.1f}  "
                          f"fill={r.get('non_stroking_color')}")

            print()

    print("=" * 70)
    print("Demo complete.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pdfplumber extraction demo")
    parser.add_argument(
        "--pdf",
        default="docs/Inv_1.pdf",
        help="Path to the PDF file (relative to src/)",
    )
    args = parser.parse_args()

    src_dir = Path(__file__).resolve().parent
    pdf_path = src_dir / args.pdf

    if not pdf_path.exists():
        print(f"ERROR: File not found: {pdf_path}")
        raise SystemExit(1)

    demo(pdf_path)
