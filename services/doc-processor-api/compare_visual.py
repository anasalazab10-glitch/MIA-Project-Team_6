"""Visual Comparison: PaddleOCR vs MinerU 2.5 on Financial PDF.

Runs both processors on PDF documents, measures execution speed, renders the pages,
draws color-coded bounding boxes for each detected element (Table, Heading, Text),
generates side-by-side comparison images, and outputs structured comparison JSON data.

Supports both:
- Parallel execution: runs MinerU and PaddleOCR concurrently side-by-side.
- Sequential execution: runs one after the other (original implementation).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont

# Add local path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.mineru_processor import MinerUDocProcessor
from app.processor import DocProcessor
from app.schemas import ContentType

# Dedicated local directory for visual comparison outputs
OUTPUT_DIR = ROOT_DIR / "visual_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def draw_bounding_boxes(
    page_pixmap: fitz.Pixmap,
    elements: list,
    target_page: int,
    model_name: str,
    output_path: Path,
    timing_sec: float | None = None,
) -> Image.Image:
    """Render page image and overlay color-coded bounding boxes for detected elements."""
    img = Image.frombytes("RGB", [page_pixmap.width, page_pixmap.height], page_pixmap.samples)
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Color mapping
    colors = {
        ContentType.TABLE: (220, 38, 38),      # Red for tables
        ContentType.HEADING: (37, 99, 235),    # Blue for headings
        ContentType.TEXT: (22, 163, 74),       # Green for text
    }

    page_elements = [e for e in elements if target_page in e.page]

    for el in page_elements:
        if not el.bbox:
            continue
        # Denormalize bbox [x0, y0, x1, y1]
        x0 = int(el.bbox[0] * w)
        y0 = int(el.bbox[1] * h)
        x1 = int(el.bbox[2] * w)
        y1 = int(el.bbox[3] * h)

        color = colors.get(el.content_type, (100, 100, 100))
        border_w = 4 if el.content_type == ContentType.TABLE else 2

        draw.rectangle([x0, y0, x1, y1], outline=color, width=border_w)

        # Label tag
        label = f"{el.content_type.value.upper()}"
        if el.content_type == ContentType.TABLE and hasattr(el.content, "rows"):
            label += f" ({len(el.content.rows)} rows)"

        # Draw small tag background
        draw.rectangle([x0, max(0, y0 - 18), x0 + len(label) * 8 + 8, y0], fill=color)
        draw.text((x0 + 4, max(0, y0 - 16)), label, fill=(255, 255, 255))

    # Add header banner
    banner_h = 44
    banner = Image.new("RGB", (w, banner_h), (30, 41, 59))
    banner_draw = ImageDraw.Draw(banner)
    
    time_str = f" | Time: {timing_sec:.2f}s" if timing_sec is not None else ""
    table_count = sum(1 for e in page_elements if e.content_type == ContentType.TABLE)
    heading_count = sum(1 for e in page_elements if e.content_type == ContentType.HEADING)
    text_count = sum(1 for e in page_elements if e.content_type == ContentType.TEXT)
    stats_str = f"Tables: {table_count}, Headings: {heading_count}, Text: {text_count}"
    
    banner_draw.text((15, 8), f"{model_name} — Page {target_page}{time_str}", fill=(255, 255, 255))
    banner_draw.text((15, 24), stats_str, fill=(203, 213, 225))

    final_img = Image.new("RGB", (w, h + banner_h))
    final_img.paste(banner, (0, 0))
    final_img.paste(img, (0, banner_h))

    final_img.save(str(output_path), "PNG")
    print(f"Saved visual representation to: {output_path.name}")
    return final_img


def create_side_by_side_image(
    page_pixmap: fitz.Pixmap,
    mineru_elements: list,
    paddle_elements: list,
    target_page: int,
    output_path: Path,
    mineru_time: float | None = None,
    paddle_time: float | None = None,
) -> None:
    """Stitches MinerU and PaddleOCR page annotations side-by-side into a single image."""
    mineru_temp = OUTPUT_DIR / f"temp_mineru_{target_page}.png"
    paddle_temp = OUTPUT_DIR / f"temp_paddle_{target_page}.png"

    img_mineru = draw_bounding_boxes(
        page_pixmap, mineru_elements, target_page, "MinerU 2.5", mineru_temp, timing_sec=mineru_time
    )
    img_paddle = draw_bounding_boxes(
        page_pixmap, paddle_elements, target_page, "PaddleOCR (PP-Structure)", paddle_temp, timing_sec=paddle_time
    )

    # Clean up temp files
    if mineru_temp.exists():
        mineru_temp.unlink()
    if paddle_temp.exists():
        paddle_temp.unlink()

    w1, h1 = img_mineru.size
    w2, h2 = img_paddle.size
    h = max(h1, h2)
    divider_w = 6

    # Canvas
    side_by_side = Image.new("RGB", (w1 + w2 + divider_w, h), (15, 23, 42))
    side_by_side.paste(img_mineru, (0, 0))
    side_by_side.paste(img_paddle, (w1 + divider_w, 0))

    side_by_side.save(str(output_path), "PNG")
    print(f"Saved SIDE-BY-SIDE visual comparison to: {output_path.name}")


def run_comparison(
    pdf_path: Path,
    parallel: bool = True,
    paddle_page_limit: int = 4,
    pages_to_render: list[int] | None = None,
) -> dict:
    """Runs MinerU and PaddleOCR with optional parallel execution."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(doc)
    print(f"\nLoaded PDF: '{pdf_path.name}' ({total_pages} pages, {len(pdf_bytes) / 1024:.1f} KB)")
    print(f"Execution Mode: {'PARALLEL (concurrent side-by-side)' if parallel else 'SEQUENTIAL'}\n")

    # Prepare sample document for PaddleOCR to avoid CPU timeouts on multi-page files
    paddle_pages_count = min(paddle_page_limit, total_pages)
    sample_doc = fitz.open()
    for pi in range(paddle_pages_count):
        sample_doc.insert_pdf(doc, from_page=pi, to_page=pi)
    paddle_bytes = sample_doc.write()
    sample_doc.close()

    mineru_proc = MinerUDocProcessor()
    paddle_proc = DocProcessor(lang="en")

    mineru_start = 0.0
    paddle_start = 0.0
    t_mineru = 0.0
    t_paddle = 0.0
    t_wall_clock = 0.0

    if parallel:
        print("=" * 70)
        print("Running MinerU 2.5 and PaddleOCR concurrently in parallel...")
        print("=" * 70)
        t0 = time.perf_counter()

        def _run_mineru():
            m_t0 = time.perf_counter()
            res = mineru_proc.process_pdf(pdf_bytes, document_id="mineru_run")
            m_dur = time.perf_counter() - m_t0
            return res, m_dur

        def _run_paddle():
            p_t0 = time.perf_counter()
            res = paddle_proc.process_pdf(paddle_bytes, document_id="paddle_run", dpi=150)
            p_dur = time.perf_counter() - p_t0
            return res, p_dur

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            fut_mineru = executor.submit(_run_mineru)
            fut_paddle = executor.submit(_run_paddle)

            (mineru_doc_id, mineru_pages, mineru_elements), t_mineru = fut_mineru.result()
            print(f"  [✓] MinerU completed in {t_mineru:.2f}s ({len(mineru_elements)} elements)")

            (paddle_doc_id, paddle_pages, paddle_elements), t_paddle = fut_paddle.result()
            print(f"  [✓] PaddleOCR completed in {t_paddle:.2f}s ({len(paddle_elements)} elements across {paddle_pages} pages)")

        t_wall_clock = time.perf_counter() - t0
    else:
        print("=" * 70)
        print("Running sequentially (original implementation)...")
        print("=" * 70)
        t0 = time.perf_counter()

        # 1. Run MinerU 2.5
        m_t0 = time.perf_counter()
        mineru_doc_id, mineru_pages, mineru_elements = mineru_proc.process_pdf(pdf_bytes, document_id="mineru_run")
        t_mineru = time.perf_counter() - m_t0
        print(f"  [✓] MinerU completed in {t_mineru:.2f}s ({len(mineru_elements)} elements)")

        # 2. Run PaddleOCR
        p_t0 = time.perf_counter()
        paddle_doc_id, paddle_pages, paddle_elements = paddle_proc.process_pdf(paddle_bytes, document_id="paddle_run", dpi=150)
        t_paddle = time.perf_counter() - p_t0
        print(f"  [✓] PaddleOCR completed in {t_paddle:.2f}s ({len(paddle_elements)} elements across {paddle_pages} pages)")

        t_wall_clock = time.perf_counter() - t0

    speedup = (t_paddle / t_mineru) if t_mineru > 0 else 1.0

    mineru_tables = [e for e in mineru_elements if e.content_type == ContentType.TABLE]
    mineru_headings = [e for e in mineru_elements if e.content_type == ContentType.HEADING]
    mineru_text = [e for e in mineru_elements if e.content_type == ContentType.TEXT]

    paddle_tables = [e for e in paddle_elements if e.content_type == ContentType.TABLE]
    paddle_headings = [e for e in paddle_elements if e.content_type == ContentType.HEADING]
    paddle_text = [e for e in paddle_elements if e.content_type == ContentType.TEXT]

    # Print Side-by-Side Comparison Table
    print("\n" + "=" * 70)
    print("SIDE-BY-SIDE BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<25} | {'MinerU 2.5':<20} | {'PaddleOCR (PP-Structure)':<25}")
    print("-" * 75)
    print(f"{'Execution Time':<25} | {f'{t_mineru:.2f}s':<20} | {f'{t_paddle:.2f}s':<25}")
    print(f"{'Speedup Factor':<25} | {f'{speedup:.1f}x faster':<20} | {'1.0x (baseline)':<25}")
    print(f"{'Pages Tested':<25} | {f'{mineru_pages} pages':<20} | {f'{paddle_pages} pages':<25}")
    print(f"{'Total Elements':<25} | {f'{len(mineru_elements)}':<20} | {f'{len(paddle_elements)}':<25}")
    print(f"{'Tables Detected':<25} | {f'{len(mineru_tables)}':<20} | {f'{len(paddle_tables)}':<25}")
    print(f"{'Headings Detected':<25} | {f'{len(mineru_headings)}':<20} | {f'{len(paddle_headings)}':<25}")
    print(f"{'Text Paragraphs':<25} | {f'{len(mineru_text)}':<20} | {f'{len(paddle_text)}':<25}")
    print(f"{'Total Wall-Clock Time':<25} | {f'{t_wall_clock:.2f}s':<20} | {'':<25}")
    print("=" * 70)

    # 3. Generate Visual Annotations
    print("\nGenerating Visual Annotations...")
    if pages_to_render is None:
        pages_to_render = [p for p in [1, 3] if p <= total_pages]
        if not pages_to_render and total_pages > 0:
            pages_to_render = [1]

    for target_p in pages_to_render:
        pix = doc[target_p - 1].get_pixmap(dpi=150)

        # Individual images
        mineru_img_path = OUTPUT_DIR / f"{pdf_path.stem}_mineru_page_{target_p}.png"
        draw_bounding_boxes(pix, mineru_elements, target_p, "MinerU 2.5", mineru_img_path, timing_sec=t_mineru)

        paddle_img_path = OUTPUT_DIR / f"{pdf_path.stem}_paddle_page_{target_p}.png"
        draw_bounding_boxes(pix, paddle_elements, target_p, "PaddleOCR (PP-Structure)", paddle_img_path, timing_sec=t_paddle)

        # Combined Side-by-Side image
        sbs_img_path = OUTPUT_DIR / f"{pdf_path.stem}_side_by_side_page_{target_p}.png"
        create_side_by_side_image(
            pix,
            mineru_elements,
            paddle_elements,
            target_p,
            sbs_img_path,
            mineru_time=t_mineru,
            paddle_time=t_paddle,
        )

    doc.close()

    # 4. Save Summary JSON
    comparison_summary = {
        "pdf_name": pdf_path.name,
        "total_pdf_pages": total_pages,
        "execution_mode": "parallel" if parallel else "sequential",
        "timing": {
            "mineru_seconds": round(t_mineru, 3),
            "paddle_seconds": round(t_paddle, 3),
            "wall_clock_seconds": round(t_wall_clock, 3),
            "speedup_factor": round(speedup, 1),
        },
        "mineru": {
            "total_elements": len(mineru_elements),
            "total_tables": len(mineru_tables),
            "total_headings": len(mineru_headings),
            "total_text": len(mineru_text),
            "tables_sample": [
                {
                    "chunk_id": t.chunk_id,
                    "page": t.page,
                    "headers": t.content.headers if hasattr(t.content, "headers") else [],
                    "num_rows": len(t.content.rows) if hasattr(t.content, "rows") else 0,
                    "sample_row": t.content.rows[0] if hasattr(t.content, "rows") and t.content.rows else [],
                }
                for t in mineru_tables[:5]
            ],
        },
        "paddle": {
            "pages_tested": paddle_pages,
            "total_elements": len(paddle_elements),
            "total_tables": len(paddle_tables),
            "total_headings": len(paddle_headings),
            "total_text": len(paddle_text),
            "tables_sample": [
                {
                    "chunk_id": t.chunk_id,
                    "page": t.page,
                    "headers": t.content.headers if hasattr(t.content, "headers") else [],
                    "num_rows": len(t.content.rows) if hasattr(t.content, "rows") else 0,
                    "sample_row": t.content.rows[0] if hasattr(t.content, "rows") and t.content.rows else [],
                }
                for t in paddle_tables[:5]
            ],
        },
    }

    summary_json_path = OUTPUT_DIR / f"{pdf_path.stem}_ocr_comparison.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_summary, f, indent=2)

    print(f"\nSaved comparison summary JSON to: {summary_json_path.name}")
    print(f"All visual images saved in: {OUTPUT_DIR.name}/")
    return comparison_summary


def find_test_pdf(requested_path: str | None = None) -> Path:
    """Find a valid PDF file to test, checking candidate test paths relative to repo."""
    if requested_path:
        p = Path(requested_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Specified PDF not found: {requested_path}")

    # Search known candidate paths relative to this script
    candidates = [
        ROOT_DIR / "test_data" / "Vodafone 2026 Annual Report.pdf",
        ROOT_DIR.parent / "raw pdf test" / "Vodafone 2026 Annual Report.pdf",
        ROOT_DIR / "test_data" / "Can_GPT_models_be_Financial_Analysts_An.pdf",
        ROOT_DIR.parent / "raw pdf test" / "Can_GPT_models_be_Financial_Analysts_An.pdf",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Search for any PDF in test_data or raw pdf test
    for search_dir in [ROOT_DIR / "test_data", ROOT_DIR.parent / "raw pdf test"]:
        if search_dir.exists():
            pdfs = sorted(search_dir.glob("*.pdf"))
            if pdfs:
                return pdfs[0]

    raise FileNotFoundError(
        "No test PDF found in default test directories. "
        "Please provide a PDF path via --pdf <path_to_pdf>"
    )


def main():
    parser = argparse.ArgumentParser(description="Side-by-side visual comparison between MinerU and PaddleOCR")
    parser.add_argument(
        "pdf_pos",
        nargs="?",
        default=None,
        help="Optional positional path to PDF file to test",
    )
    parser.add_argument(
        "--pdf",
        dest="pdf_opt",
        type=str,
        default=None,
        help="Path to PDF file to test (defaults to test_data/Vodafone 2026 Annual Report.pdf)",
    )
    parser.add_argument(
        "--parallel",
        dest="parallel",
        action="store_true",
        default=True,
        help="Run MinerU and PaddleOCR in parallel concurrently (default)",
    )
    parser.add_argument(
        "--sequential",
        dest="parallel",
        action="store_false",
        help="Run MinerU and PaddleOCR sequentially (original implementation)",
    )
    parser.add_argument(
        "--paddle-pages",
        type=int,
        default=4,
        help="Maximum pages to process with PaddleOCR to protect CPU/memory (default: 4)",
    )
    args = parser.parse_args()

    chosen_pdf_arg = args.pdf_opt or args.pdf_pos
    pdf_path = find_test_pdf(chosen_pdf_arg)

    run_comparison(pdf_path, parallel=args.parallel, paddle_page_limit=args.paddle_pages)


if __name__ == "__main__":
    main()
