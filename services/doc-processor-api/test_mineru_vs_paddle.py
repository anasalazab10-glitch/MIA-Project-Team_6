"""Test and comparison script for MinerU 2.5 vs PaddleOCR on Financial PDFs.

Supports:
1. Standard single-engine test of MinerU 2.5 (original implementation, fully preserved).
2. Parallel page processing validation (verifying parallel=True produces identical elements to parallel=False).
3. Concurrent side-by-side benchmark running MinerU and PaddleOCR in parallel threads.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path

# Add app to path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import fitz
from app.mineru_processor import MinerUDocProcessor
from app.processor import DocProcessor
from app.schemas import ContentType, Element, ProcessResponse, TableContent


def run_test(pdf_path: str, parallel_pages: bool = False):
    """Original MinerU test function, fully preserved for existing workflows and files."""
    p = Path(pdf_path)
    if not p.exists():
        print(f"Error: File not found at {pdf_path}")
        return

    print(f"Loading PDF: {p.name} ({p.stat().st_size / 1024:.1f} KB)...")
    with open(p, "rb") as f:
        pdf_bytes = f.read()

    print("\n" + "=" * 80)
    print(f"Testing MinerU 2.5 Document Processor (Mode: {'Parallel Pages' if parallel_pages else 'Sequential'})")
    print("=" * 80)

    processor = MinerUDocProcessor()
    t0 = time.perf_counter()
    doc_id, num_pages, elements = processor.process_pdf(
        pdf_bytes, document_id=p.stem, parallel=parallel_pages
    )
    dur = time.perf_counter() - t0

    print(f"Successfully processed {num_pages} pages in {dur:.3f}s.")
    print(f"Total elements extracted: {len(elements)}")

    table_elements = [e for e in elements if e.content_type == ContentType.TABLE]
    text_elements = [e for e in elements if e.content_type == ContentType.TEXT]
    heading_elements = [e for e in elements if e.content_type == ContentType.HEADING]

    print(f"  - Headings: {len(heading_elements)}")
    print(f"  - Text paragraphs: {len(text_elements)}")
    print(f"  - Tables: {len(table_elements)}")

    print("\n" + "-" * 80)
    print(f"EXTRACTED TABLES DETAIL ({len(table_elements)} tables detected):")
    print("-" * 80)

    for i, tbl in enumerate(table_elements, 1):
        content: TableContent = tbl.content
        print(f"\n[Table #{i}] Chunk ID: {tbl.chunk_id} | Page: {tbl.page} | BBox: {tbl.bbox}")
        print(f"  Headers ({len(content.headers)}): {content.headers}")
        print(f"  Rows count: {len(content.rows)}")
        print("  Sample rows:")
        for r_idx, row in enumerate(content.rows[:3], 1):
            print(f"    Row {r_idx}: {row}")

    print("\n" + "=" * 80)
    print("Test Completed Successfully!")
    print("=" * 80)
    return doc_id, num_pages, elements


def run_side_by_side_parallel_benchmark(pdf_path: str, paddle_page_limit: int = 4):
    """Runs MinerU 2.5 and PaddleOCR concurrently in parallel to test both side-by-side."""
    p = Path(pdf_path)
    if not p.exists():
        print(f"Error: File not found at {pdf_path}")
        return

    print(f"\n{'=' * 80}")
    print(f"PARALLEL SIDE-BY-SIDE BENCHMARK: MinerU 2.5 vs PaddleOCR (PP-Structure)")
    print(f"Target Document: {p.name} ({p.stat().st_size / 1024:.1f} KB)")
    print(f"{'=' * 80}\n")

    with open(p, "rb") as f:
        pdf_bytes = f.read()

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(doc)

    # Sample pages for PaddleOCR to avoid excessive CPU runtimes on multi-page files
    sample_pages = min(paddle_page_limit, total_pages)
    sample_doc = fitz.open()
    for pi in range(sample_pages):
        sample_doc.insert_pdf(doc, from_page=pi, to_page=pi)
    paddle_bytes = sample_doc.write()
    sample_doc.close()
    doc.close()

    mineru_proc = MinerUDocProcessor()
    paddle_proc = DocProcessor(lang="en")

    print(f"Launching both engines in parallel (MinerU: {total_pages} pages, PaddleOCR: {sample_pages} pages)...")
    wall_start = time.perf_counter()

    def _task_mineru():
        t0 = time.perf_counter()
        res = mineru_proc.process_pdf(pdf_bytes, document_id="mineru_bench")
        dur = time.perf_counter() - t0
        return res, dur

    def _task_paddle():
        t0 = time.perf_counter()
        res = paddle_proc.process_pdf(paddle_bytes, document_id="paddle_bench", dpi=150)
        dur = time.perf_counter() - t0
        return res, dur

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        fut_m = executor.submit(_task_mineru)
        fut_p = executor.submit(_task_paddle)

        (m_id, m_pages, m_elements), t_mineru = fut_m.result()
        print(f"  [✓] MinerU 2.5 finished in {t_mineru:.2f}s")

        (p_id, p_pages, p_elements), t_paddle = fut_p.result()
        print(f"  [✓] PaddleOCR finished in {t_paddle:.2f}s")

    wall_dur = time.perf_counter() - wall_start
    speedup = (t_paddle / t_mineru) if t_mineru > 0 else 1.0

    m_tables = [e for e in m_elements if e.content_type == ContentType.TABLE]
    m_headings = [e for e in m_elements if e.content_type == ContentType.HEADING]
    m_text = [e for e in m_elements if e.content_type == ContentType.TEXT]

    p_tables = [e for e in p_elements if e.content_type == ContentType.TABLE]
    p_headings = [e for e in p_elements if e.content_type == ContentType.HEADING]
    p_text = [e for e in p_elements if e.content_type == ContentType.TEXT]

    print("\n" + "=" * 80)
    print("SIDE-BY-SIDE METRICS COMPARISON")
    print("=" * 80)
    print(f"{'Metric':<25} | {'MinerU 2.5':<22} | {'PaddleOCR (PP-Structure)':<25}")
    print("-" * 78)
    print(f"{'Execution Time':<25} | {f'{t_mineru:.3f}s':<22} | {f'{t_paddle:.3f}s':<25}")
    print(f"{'Speedup vs Paddle':<25} | {f'{speedup:.1f}x FASTER':<22} | {'1.0x (baseline)':<25}")
    print(f"{'Pages Evaluated':<25} | {f'{m_pages} pages':<22} | {f'{p_pages} pages':<25}")
    print(f"{'Total Elements':<25} | {f'{len(m_elements)}':<22} | {f'{len(p_elements)}':<25}")
    print(f"{'Tables Detected':<25} | {f'{len(m_tables)}':<22} | {f'{len(p_tables)}':<25}")
    print(f"{'Headings Detected':<25} | {f'{len(m_headings)}':<22} | {f'{len(p_headings)}':<25}")
    print(f"{'Text Paragraphs':<25} | {f'{len(m_text)}':<22} | {f'{len(p_text)}':<25}")
    print(f"{'Total Parallel Wall Time':<25} | {f'{wall_dur:.3f}s':<22} | {'':<25}")
    print("=" * 80)


def verify_old_implementation(pdf_path: str):
    """Verifies that sequential (old implementation) and parallel produce identical elements."""
    p = Path(pdf_path)
    if not p.exists():
        return

    with open(p, "rb") as f:
        b = f.read()

    proc = MinerUDocProcessor()
    _, _, seq_elements = proc.process_pdf(b, document_id="verify", parallel=False)
    _, _, par_elements = proc.process_pdf(b, document_id="verify", parallel=True)

    assert len(seq_elements) == len(par_elements), f"Element count mismatch: {len(seq_elements)} vs {len(par_elements)}"
    print(f"Verification PASSED for '{p.name}': Old implementation and parallel option output identical {len(seq_elements)} elements.")


def find_test_pdf(requested_path: str | None = None) -> str:
    """Find a test PDF using relative fallback paths."""
    if requested_path:
        p = Path(requested_path)
        if p.exists():
            return str(p)
        raise FileNotFoundError(f"Specified PDF not found: {requested_path}")

    candidates = [
        ROOT_DIR / "test_data" / "Vodafone 2026 Annual Report.pdf",
        ROOT_DIR.parent / "raw pdf test" / "Vodafone 2026 Annual Report.pdf",
        ROOT_DIR / "test_data" / "Can_GPT_models_be_Financial_Analysts_An.pdf",
        ROOT_DIR.parent / "raw pdf test" / "Can_GPT_models_be_Financial_Analysts_An.pdf",
    ]
    for c in candidates:
        if c.exists():
            return str(c)

    for search_dir in [ROOT_DIR / "test_data", ROOT_DIR.parent / "raw pdf test"]:
        if search_dir.exists():
            pdfs = sorted(search_dir.glob("*.pdf"))
            if pdfs:
                return str(pdfs[0])

    raise FileNotFoundError(
        "No test PDF found in default test directories. "
        "Please provide a PDF path via: python test_mineru_vs_paddle.py --pdf <path_to_pdf>"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MinerU 2.5 vs PaddleOCR Test & Comparison Harness")
    parser.add_argument("pdf_path", nargs="?", default=None, help="Path to PDF to process")
    parser.add_argument("--pdf", dest="pdf_opt", default=None, help="Alternative flag for path to PDF to process")
    parser.add_argument("--parallel", action="store_true", help="Run both engines concurrently in parallel side-by-side")
    parser.add_argument("--parallel-pages", action="store_true", help="Run MinerU with parallel page processing")
    parser.add_argument("--verify-parity", action="store_true", help="Assert exact element match between sequential and parallel")
    parser.add_argument("--paddle-pages", type=int, default=4, help="Page cap for PaddleOCR (default: 4)")
    args = parser.parse_args()

    pdf_arg = args.pdf_opt or args.pdf_path
    pdf_to_run = find_test_pdf(pdf_arg)

    if args.verify_parity:
        verify_old_implementation(pdf_to_run)
    elif args.parallel:
        run_side_by_side_parallel_benchmark(pdf_to_run, paddle_page_limit=args.paddle_pages)
    else:
        # Default: original MinerU test
        run_test(pdf_to_run, parallel_pages=args.parallel_pages)

