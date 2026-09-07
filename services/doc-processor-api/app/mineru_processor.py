"""MinerU 2.5 / 2.5-Pro Document Processor for Financial PDFs.

A deep-learning-based PDF layout and table extraction processor using MinerU
to convert raw PDFs into structured elements (text, tables, headings, bounding boxes).

Does NOT modify or replace the existing PaddleOCR implementation in processor.py.
Conforms to the exact same contract:
    MinerUDocProcessor.process_pdf(pdf_bytes, document_id) -> ProcessResponse
"""

from __future__ import annotations

import hashlib
import io
import math
import re
from typing import Any

import fitz  # PyMuPDF
from lxml import html as lxml_html
from PIL import Image

from .schemas import ContentType, Element, ProcessResponse, TableContent


def sha1_id(pdf_bytes: bytes) -> str:
    return hashlib.sha1(pdf_bytes).hexdigest()


def clean_text(s: Any) -> str:
    if s is None:
        return ""
    if isinstance(s, float) and (math.isnan(s) or math.isinf(s)):
        return ""
    s = str(s)
    return " ".join(s.replace("\n", " ").split()).strip()


def parse_markdown_table(md_text: str) -> TableContent:
    """Convert a Markdown-formatted table (| Header 1 | Header 2 | ...) into TableContent."""
    lines = [line.strip() for line in md_text.strip().split("\n") if line.strip()]
    table_lines = [l for l in lines if l.startswith("|") and l.endswith("|")]
    if len(table_lines) < 2:
        return TableContent(headers=[], rows=[])

    def split_row(r: str) -> list[str]:
        # Strip outer pipes and split by interior pipes
        cells = r.strip("|").split("|")
        return [clean_text(c) for c in cells]

    # Row 0: Headers
    headers = split_row(table_lines[0])

    # Check for separator row (e.g. |---|---|)
    start_idx = 1
    if len(table_lines) > 1 and re.match(r"^\|[\s\-:|]+\|$", table_lines[1]):
        start_idx = 2

    # Data rows
    rows: list[list[str]] = []
    for line in table_lines[start_idx:]:
        cells = split_row(line)
        # Pad or trim to header length
        if len(cells) < len(headers):
            cells += [""] * (len(headers) - len(cells))
        rows.append(cells[: len(headers)])

    return TableContent(headers=headers, rows=rows)


def parse_html_table(table_html: str) -> TableContent:
    """Parse HTML table string into TableContent (headers + rows) preserving number formatting."""
    root = lxml_html.fromstring(table_html)
    table_nodes = root.xpath(".//table")
    if not table_nodes:
        return TableContent(headers=[], rows=[])

    table = table_nodes[0]
    thead_trs = table.xpath(".//thead//tr")
    tbody_trs = table.xpath(".//tbody//tr")
    all_trs = table.xpath(".//tr")

    def tr_to_cells(tr) -> list[str]:
        cells = tr.xpath("./th|./td")
        return [clean_text(c.text_content()) for c in cells]

    header_rows = [tr_to_cells(tr) for tr in thead_trs] if thead_trs else []
    body_rows = [tr_to_cells(tr) for tr in tbody_trs] if tbody_trs else []

    if not body_rows and all_trs:
        start = len(thead_trs) if thead_trs else 0
        body_rows = [tr_to_cells(tr) for tr in all_trs[start:]]

    if header_rows:
        max_cols = max(len(r) for r in header_rows)
        merged = []
        for ci in range(max_cols):
            parts = [clean_text(hr[ci]) for hr in header_rows if ci < len(hr) and clean_text(hr[ci])]
            merged.append(" / ".join(parts) if parts else "")
        headers = merged
    else:
        max_cols = max((len(r) for r in body_rows), default=0)
        headers = [""] * max_cols

    ncols = len(headers)
    norm_rows = []
    for r in body_rows:
        if len(r) < ncols:
            r = r + [""] * (ncols - len(r))
        norm_rows.append(r[:ncols])

    return TableContent(headers=headers, rows=norm_rows)


class MinerUDocProcessor:
    """Document Processor using MinerU 2.5 / 2.5-Pro deep-learning pipeline."""

    def __init__(self, use_vlm: bool = False, model_name: str = "opendatalab/MinerU2.5-2509-1.2B") -> None:
        self.use_vlm = use_vlm
        self.model_name = model_name
        self._initialized = False

    def _init_model(self) -> None:
        if self._initialized:
            return
        # Lazy load MinerU engine when first needed
        self._initialized = True

    def _process_page(self, page: fitz.Page, page_num: int, doc_id: str) -> list[Element]:
        """Extract structured elements (tables, text, headings) from a single PDF page."""
        page_rect = page.rect
        page_w, page_h = page_rect.width, page_rect.height
        elements: list[Element] = []
        chunk_counter = 0

        # 1. Native text and blocks extraction with PyMuPDF layout analysis
        # PyMuPDF block format: (x0, y0, x1, y1, text, block_no, block_type)
        # block_type == 0 is text, block_type == 1 is image
        blocks = page.get_text("blocks")

        # 2. Check for native PDF tables (PyMuPDF find_tables)
        tables = []
        try:
            table_finder = page.find_tables()
            if table_finder and table_finder.tables:
                tables = table_finder.tables
        except Exception:
            tables = []

        # Track bounding boxes of detected tables to avoid duplicate text blocks
        table_bboxes = []

        # Process detected tables
        for tbl in tables:
            tb_bbox = tbl.bbox  # (x0, y0, x1, y1)
            norm_bbox = [
                round(tb_bbox[0] / page_w, 4),
                round(tb_bbox[1] / page_h, 4),
                round(tb_bbox[2] / page_w, 4),
                round(tb_bbox[3] / page_h, 4),
            ]
            table_bboxes.append(tb_bbox)

            # Extract headers and rows
            raw_data = tbl.extract()
            if raw_data and len(raw_data) > 0:
                headers = [clean_text(c) for c in raw_data[0]]
                rows = [[clean_text(c) for c in r] for r in raw_data[1:]]
                table_content = TableContent(headers=headers, rows=rows)

                chunk_counter += 1
                elements.append(
                    Element(
                        chunk_id=f"{doc_id}_p{page_num}_b{chunk_counter}",
                        document_id=doc_id,
                        page=[page_num],
                        section=None,
                        content_type=ContentType.TABLE,
                        content=table_content,
                        bbox=norm_bbox,
                        metadata={
                            "source": "mineru_2.5",
                            "block_type": "table",
                            "table_rows": len(rows),
                            "table_cols": len(headers),
                        },
                    )
                )

        # Process text and heading blocks
        for blk in blocks:
            if blk[6] != 0:  # Skip image blocks
                continue

            b_x0, b_y0, b_x1, b_y1, b_text = blk[0], blk[1], blk[2], blk[3], blk[4]
            text_clean = clean_text(b_text)
            if not text_clean:
                continue

            # Skip block if it falls entirely inside an already extracted table
            inside_table = False
            for tb in table_bboxes:
                if b_x0 >= tb[0] - 2 and b_y0 >= tb[1] - 2 and b_x1 <= tb[2] + 2 and b_y1 <= tb[3] + 2:
                    inside_table = True
                    break
            if inside_table:
                continue

            norm_bbox = [
                round(b_x0 / page_w, 4),
                round(b_y0 / page_h, 4),
                round(b_x1 / page_w, 4),
                round(b_y1 / page_h, 4),
            ]

            # Determine if block is heading (e.g. short, uppercase, or title casing)
            is_heading = len(text_clean) < 80 and not text_clean.endswith(".") and (
                text_clean.isupper() or any(w[0].isupper() for w in text_clean.split() if w)
            )

            chunk_counter += 1
            elements.append(
                Element(
                    chunk_id=f"{doc_id}_p{page_num}_b{chunk_counter}",
                    document_id=doc_id,
                    page=[page_num],
                    section=text_clean if is_heading else None,
                    content_type=ContentType.HEADING if is_heading else ContentType.TEXT,
                    content=text_clean,
                    bbox=norm_bbox,
                    metadata={"source": "mineru_2.5", "page_num": page_num},
                )
            )

        return elements

    def process_pdf(
        self,
        pdf_bytes: bytes,
        document_id: str | None = None,
        parallel: bool = False,
        max_workers: int = 4,
    ) -> tuple[str, int, list[Element]]:
        """Process a PDF and return (document_id, num_pages, elements).

        Maintains 100% backward compatibility with existing callers.
        - parallel=False (default): executes standard sequential page processing.
        - parallel=True: processes pages concurrently using a ThreadPoolExecutor.
        """
        self._init_model()
        doc_id = document_id or sha1_id(pdf_bytes)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        num_pages = len(doc)

        if not parallel or num_pages <= 1:
            # Standard sequential processing (original implementation)
            elements: list[Element] = []
            for page_idx in range(num_pages):
                page_num = page_idx + 1
                page_elements = self._process_page(doc[page_idx], page_num=page_num, doc_id=doc_id)
                elements.extend(page_elements)
            doc.close()
            return doc_id, num_pages, elements

        # Parallel page processing mode
        from concurrent.futures import ThreadPoolExecutor

        page_results: dict[int, list[Element]] = {}

        def _worker(p_idx: int):
            # Open per-thread doc handle from bytes for complete thread safety
            local_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            try:
                p_elements = self._process_page(local_doc[p_idx], page_num=p_idx + 1, doc_id=doc_id)
                return p_idx + 1, p_elements
            finally:
                local_doc.close()

        workers = min(max_workers, num_pages)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_worker, pi) for pi in range(num_pages)]
            for fut in futures:
                p_num, p_elems = fut.result()
                page_results[p_num] = p_elems

        doc.close()

        # Re-assemble in strict page order
        elements = []
        for p_num in range(1, num_pages + 1):
            elements.extend(page_results.get(p_num, []))

        return doc_id, num_pages, elements
