from __future__ import annotations

import hashlib
import math
import os
import re
import threading
from typing import Any

import fitz  # PyMuPDF
import numpy as np
from lxml import html as lxml_html
from PIL import Image
from paddleocr import PaddleOCR, PPStructure

from .schemas import ContentType, Element, TableContent


def sha1_id(pdf_bytes: bytes) -> str:
    return hashlib.sha1(pdf_bytes).hexdigest()


def normalize_bbox(bbox: list[float] | None, w: int, h: int) -> list[float] | None:
    if not bbox or w <= 0 or h <= 0:
        return None
    x0, y0, x1, y1 = bbox
    return [x0 / w, y0 / h, x1 / w, y1 / h]


def clean_text(s: Any) -> str:
    if s is None:
        return ""
    if isinstance(s, float) and (math.isnan(s) or math.isinf(s)):
        return ""
    s = str(s)
    return " ".join(s.replace("\n", " ").split()).strip()



def _bbox_height_px(b: dict[str, Any]) -> float:
    bb = b.get("bbox")
    if not bb or len(bb) != 4:
        return 0.0
    return float(bb[3] - bb[1])

_HEADING_RE = re.compile(r"^(\d{1,2}[\.|\)]\s+.+|[A-Z][A-Za-z0-9&\-/ ]{2,})$")

def looks_like_heading(text: str, bbox_h: float, median_text_h: float) -> bool:
    t = clean_text(text)
    if not t:
        return False
    # Too long is rarely a heading
    if len(t) > 80:
        return False
    # Common heading patterns: "3. Debtors", "2) Fixed assets", or Title-ish short text
    regex_ok = bool(_HEADING_RE.match(t)) and not t.endswith(".")
    # Tall text relative to typical text lines on the page
    tall_ok = (median_text_h > 0) and (bbox_h >= 1.6 * median_text_h) and (len(t) <= 60)
    # Avoid promoting sentences
    sentence_like = t.endswith(".") or t.count(" ") > 12
    return (regex_ok or tall_ok) and not sentence_like

def extract_text(res_field: Any) -> str:
    if res_field is None:
        return ""
    if isinstance(res_field, str):
        return clean_text(res_field)
    if isinstance(res_field, list):
        parts: list[str] = []
        for item in res_field:
            if isinstance(item, dict) and "text" in item:
                t = clean_text(item["text"])
                if t:
                    parts.append(t)
            else:
                t = clean_text(item)
                if t:
                    parts.append(t)
        return "\n".join(parts).strip()
    if isinstance(res_field, dict) and "text" in res_field:
        return clean_text(res_field["text"])
    return clean_text(res_field)


def _int_attr(cell, name: str) -> int:
    raw = cell.get(name)
    try:
        n = int(raw) if raw else 1
    except ValueError:
        n = 1
    return max(n, 1)


def _rows_to_grid(trs: list[Any]) -> list[list[str]]:
    """
    Turns a list of <tr> elements into a rectangular grid of strings,
    correctly honoring BOTH colspan and rowspan. A rowspan cell is
    "remembered" in `pending` and re-emitted into the same column index
    on the following rows until it's exhausted, so row labels/values
    never silently shift left the way a colspan-only implementation
    would cause.
    """
    grid: list[list[str]] = []
    # col_index -> (text, rows_remaining)
    pending: dict[int, tuple[str, int]] = {}

    for tr in trs:
        row: list[str] = []
        col = 0
        cells = iter(tr.xpath("./th|./td"))
        cell = next(cells, None)

        while cell is not None or col in pending:
            if col in pending:
                text, remaining = pending[col]
                row.append(text)
                pending[col] = (text, remaining - 1) if remaining > 1 else None
                if pending[col] is None:
                    del pending[col]
                col += 1
                continue

            text = clean_text(cell.text_content())
            colspan = _int_attr(cell, "colspan")
            rowspan = _int_attr(cell, "rowspan")

            for _ in range(colspan):
                row.append(text)
                if rowspan > 1:
                    pending[col] = (text, rowspan - 1)
                col += 1

            cell = next(cells, None)

        grid.append(row)

    max_cols = max((len(r) for r in grid), default=0)
    for r in grid:
        if len(r) < max_cols:
            r.extend([""] * (max_cols - len(r)))

    return grid


def html_table_to_content(table_html: str) -> TableContent:
    if not table_html or not table_html.strip():
        return TableContent(headers=[], rows=[])

    root = lxml_html.fromstring(table_html)
    table_nodes = root.xpath("descendant-or-self::table")
    if not table_nodes:
        return TableContent(headers=[], rows=[])

    table = table_nodes[0]

    thead_trs = table.xpath(".//thead//tr")
    tbody_trs = table.xpath(".//tbody//tr")
    all_trs = table.xpath(".//tr")

    header_rows = _rows_to_grid(thead_trs) if thead_trs else []
    body_rows = _rows_to_grid(tbody_trs) if tbody_trs else []

    if not body_rows and all_trs:
        start = len(thead_trs) if thead_trs else 0
        body_rows = _rows_to_grid(all_trs[start:])

    if header_rows:
        max_cols = max(len(r) for r in header_rows)
        merged: list[str] = []
        for ci in range(max_cols):
            parts = []
            for hr in header_rows:
                if ci < len(hr):
                    v = clean_text(hr[ci])
                    if v:
                        parts.append(v)
            merged.append(" / ".join(parts) if parts else "")
        headers = merged
    else:
        max_cols = max((len(r) for r in body_rows), default=0)
        headers = [""] * max_cols

    ncols = len(headers)
    norm_rows: list[list[str]] = []
    for r in body_rows:
        r = [clean_text(x) for x in r]
        if len(r) < ncols:
            r = r + [""] * (ncols - len(r))
        elif len(r) > ncols:
            headers = headers + [""] * (len(r) - ncols)
            ncols = len(headers)
        norm_rows.append(r)

    return TableContent(headers=[clean_text(h) for h in headers], rows=norm_rows)


def table_to_row_facts(table: TableContent, section: str | None = None) -> list[str]:
    """
    Serializes a table into one self-contained sentence per non-empty cell,
    This keeps the row-label <-> column-header <-> value relationship
    explicit even after the table leaves this service, so a downstream
    retriever/embedder doesn't have to reconstruct alignment from position.
    Assumes the first column of each row is the row label (true for the
    vast majority of financial statement tables coming out of PPStructure).
    """
    if not table.rows:
        return []

    facts: list[str] = []
    prefix = f"{section}. " if section else ""

    for row in table.rows:
        if not row:
            continue
        row_label = row[0].strip()
        for col_idx in range(1, len(row)):
            value = row[col_idx].strip()
            if not value:
                continue
            col_header = table.headers[col_idx].strip() if col_idx < len(table.headers) else ""
            if row_label and col_header:
                facts.append(f"{prefix}{row_label} ({col_header}): {value}")
            elif row_label:
                facts.append(f"{prefix}{row_label}: {value}")
            else:
                facts.append(f"{prefix}{col_header}: {value}")

    return facts


def page_ocr_to_text(ocr_result: Any) -> str:
    """
    PaddleOCR.ocr(image) returns something like:
    [ [ [box_pts], (text, score) ], ... ]
    We sort roughly top-to-bottom, left-to-right.
    """
    if not ocr_result:
        return ""

    # sometimes it's wrapped: [results_for_image]
    if isinstance(ocr_result, list) and len(ocr_result) == 1 and isinstance(ocr_result[0], list):
        lines = ocr_result[0]
    else:
        lines = ocr_result

    extracted = []
    for line in lines:
        try:
            box, (text, score) = line
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            x0, y0 = min(xs), min(ys)
            extracted.append((y0, x0, clean_text(text)))
        except Exception:
            continue

    extracted.sort(key=lambda t: (t[0], t[1]))
    return "\n".join([t[2] for t in extracted if t[2]]).strip()


class DocProcessor:
    def __init__(self, lang: str = "en"):
        self.lang = lang
        self._engine = PPStructure(
            lang=lang,
            show_log=False,
            recovery=True,
            table=True,
            ocr=True,
        )
        # Full-page OCR fallback (only used when include_full_page_ocr=True)
        self._page_ocr = PaddleOCR(lang=lang, show_log=False, use_angle_cls=False)
        self._lock = threading.Lock()

    def process_pdf(
        self,
        pdf_bytes: bytes,
        document_id: str | None = None,
        dpi: int = 200,
        include_full_page_ocr: bool = False,
    ) -> tuple[str, int, list[Element]]:
        doc_id = document_id or sha1_id(pdf_bytes)

        with self._lock:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")

            requested_dpi = dpi
            if doc.page_count > 1:
                cap = int(os.getenv("DOC_PROCESSOR_MAX_DPI_MULTIPAGE", "120"))
                dpi = min(dpi, cap)

            all_elements: list[Element] = []

            for page_index in range(doc.page_count):
                page_num = page_index + 1

                page = doc[page_index]
                pix = page.get_pixmap(dpi=dpi)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                w, h = img.size
                np_img = np.array(img)[:, :, ::-1]  # RGB->BGR

                blocks = self._engine(np_img)

                def key_fn(b: dict[str, Any]) -> tuple[float, float]:
                    bb = b.get("bbox") or [0, 0, 0, 0]
                    return (bb[1], bb[0])

                blocks = sorted(blocks, key=key_fn)

                # Compute median text bbox height (px) for heading-promotion heuristic

                text_heights = [_bbox_height_px(x) for x in blocks if x.get('type') == 'text']

                text_heights = [h for h in text_heights if h > 0]

                text_heights.sort()

                median_text_h = text_heights[len(text_heights)//2] if text_heights else 0.0

                current_section: str | None = None
                current_section_chunk_id: str | None = None

                for bi, b in enumerate(blocks):
                    btype = b.get("type")
                    bbox_norm = normalize_bbox(b.get("bbox"), w, h)

                    if btype == "title":
                        content_type = ContentType.HEADING
                    elif btype == "table":
                        content_type = ContentType.TABLE
                    else:
                        content_type = ContentType.TEXT

                    # heading promotion flag (must exist for ALL block types, including tables)
                    heading_promoted = False

                    if content_type == ContentType.TABLE:
                        html = (b.get("res") or {}).get("html", "")
                        content = html_table_to_content(html) if html else TableContent(headers=[], rows=[])
                    else:
                        content = extract_text(b.get("res"))

                    heading_promoted = False
                    if content_type == ContentType.TEXT and btype == "text" and isinstance(content, str):
                        if looks_like_heading(content, _bbox_height_px(b), median_text_h):
                            content_type = ContentType.HEADING
                            heading_promoted = True

                    chunk_id = f"{doc_id}_p{page_num}_b{bi}"

                    if content_type == ContentType.HEADING:
                        if isinstance(content, str) and content.strip():
                            current_section = content.strip()
                            current_section_chunk_id = chunk_id
                        section = current_section
                    else:
                        section = current_section

                    element_metadata: dict[str, Any] = {
                        "source": "paddleocr_ppstructure",
                        "block_type": btype,
                        "bbox_unit": "normalized",
                        "render_dpi": dpi,
                        "requested_dpi": requested_dpi,
                        "dpi_capped": dpi != requested_dpi,
                        "page_image_size": [w, h],
                        "heading_promoted": heading_promoted,
                        "section_anchor_chunk_id": current_section_chunk_id,
                    }

                    if content_type == ContentType.TABLE and isinstance(content, TableContent):
                        element_metadata["table_row_facts"] = table_to_row_facts(content, section)

                    all_elements.append(
                        Element(
                            chunk_id=chunk_id,
                            document_id=doc_id,
                            page=[page_num],
                            section=section,
                            content_type=content_type,
                            content=content,
                            bbox=bbox_norm,
                            metadata=element_metadata,
                        )
                    )

                if include_full_page_ocr:
                    ocr_res = self._page_ocr.ocr(np_img, cls=False)
                    page_text = page_ocr_to_text(ocr_res)

                    all_elements.append(
                        Element(
                            chunk_id=f"{doc_id}_p{page_num}_fullocr",
                            document_id=doc_id,
                            page=[page_num],
                            section=None,
                            content_type=ContentType.TEXT,
                            content=page_text,
                            bbox=None,
                            metadata={
                                "source": "paddleocr_full_page_ocr",
                                "full_page_ocr": True,
                                "render_dpi": dpi,
                                "requested_dpi": requested_dpi,
                                "dpi_capped": dpi != requested_dpi,
                                "page_image_size": [w, h],
                            },
                        )
                    )

        return doc_id, doc.page_count, all_elements
