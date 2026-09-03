from __future__ import annotations

import hashlib
import math
import threading
from typing import Any

import fitz  # PyMuPDF
import numpy as np
from lxml import html as lxml_html
from PIL import Image
from paddleocr import PPStructure

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


def extract_text(res_field: Any) -> str:
    """
    PaddleOCR may output:
    - str
    - list[dict] with {"text": "..."}
    - list[str]
    """
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


def _expand_colspan(cell) -> list[str]:
    text = clean_text(cell.text_content())
    colspan = cell.get("colspan")
    try:
        n = int(colspan) if colspan else 1
    except ValueError:
        n = 1
    return [text] * max(n, 1)


def _tr_to_cells(tr) -> list[str]:
    # include both th and td
    cells = tr.xpath("./th|./td")
    out: list[str] = []
    for c in cells:
        out.extend(_expand_colspan(c))
    return out


def html_table_to_content(table_html: str) -> TableContent:
    """
    Parse PaddleOCR table HTML into headers+rows while preserving formatting
    (e.g., commas in numbers) and avoiding pandas type coercion.
    """
    root = lxml_html.fromstring(table_html)
    table_nodes = root.xpath(".//table")
    if not table_nodes:
        return TableContent(headers=[], rows=[])

    table = table_nodes[0]

    thead_trs = table.xpath(".//thead//tr")
    tbody_trs = table.xpath(".//tbody//tr")
    all_trs = table.xpath(".//tr")

    header_rows = [_tr_to_cells(tr) for tr in thead_trs] if thead_trs else []
    body_rows = [_tr_to_cells(tr) for tr in tbody_trs] if tbody_trs else []

    # fallback if no tbody
    if not body_rows and all_trs:
        # if we had headers, skip those rows; else treat all as body
        start = len(thead_trs) if thead_trs else 0
        body_rows = [_tr_to_cells(tr) for tr in all_trs[start:]]

    # determine headers
    if header_rows:
        # If multiple header rows, merge column-wise: "A / B"
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
        # no header -> empty headers sized to max columns in body
        max_cols = max((len(r) for r in body_rows), default=0)
        headers = [""] * max_cols

    # Normalize row lengths to header length
    ncols = len(headers)
    norm_rows: list[list[str]] = []
    for r in body_rows:
        r = [clean_text(x) for x in r]
        if len(r) < ncols:
            r = r + [""] * (ncols - len(r))
        elif len(r) > ncols:
            # extend headers if row longer
            headers = headers + [""] * (len(r) - ncols)
            ncols = len(headers)
        norm_rows.append(r)

    return TableContent(headers=[clean_text(h) for h in headers], rows=norm_rows)


class DocProcessor:
    """
    Holds a single PPStructure engine instance.
    Locking avoids thread-safety issues with underlying predictors.
    """

    def __init__(self, lang: str = "en"):
        self.lang = lang
        self._engine = PPStructure(
            lang=lang,
            show_log=False,
            recovery=True,
            table=True,
            ocr=True,
        )
        self._lock = threading.Lock()

    def process_pdf(
        self,
        pdf_bytes: bytes,
        document_id: str | None = None,
        dpi: int = 200,
    ) -> tuple[str, int, list[Element]]:
        doc_id = document_id or sha1_id(pdf_bytes)

        with self._lock:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            all_elements: list[Element] = []

            for page_index in range(doc.page_count):
                page_num = page_index + 1  # 1-based output

                page = doc[page_index]
                pix = page.get_pixmap(dpi=dpi)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                w, h = img.size
                np_img = np.array(img)[:, :, ::-1]  # RGB->BGR

                blocks = self._engine(np_img)

                # Sort blocks top-to-bottom, left-to-right for consistent section assignment
                def key_fn(b: dict[str, Any]) -> tuple[float, float]:
                    bb = b.get("bbox") or [0, 0, 0, 0]
                    return (bb[1], bb[0])

                blocks = sorted(blocks, key=key_fn)

                current_section: str | None = None

                for bi, b in enumerate(blocks):
                    btype = b.get("type")
                    bbox_norm = normalize_bbox(b.get("bbox"), w, h)

                    if btype == "title":
                        content_type = ContentType.HEADING
                    elif btype == "table":
                        content_type = ContentType.TABLE
                    else:
                        content_type = ContentType.TEXT

                    if content_type == ContentType.TABLE:
                        html = (b.get("res") or {}).get("html", "")
                        content = html_table_to_content(html) if html else TableContent(headers=[], rows=[])
                    else:
                        content = extract_text(b.get("res"))

                    # Nearest-title-above section rule
                    if content_type == ContentType.HEADING:
                        if isinstance(content, str) and content.strip():
                            current_section = content.strip()
                        section = current_section
                    else:
                        section = current_section

                    el = Element(
                        chunk_id=f"{doc_id}_p{page_num}_b{bi}",
                        document_id=doc_id,
                        page=[page_num],
                        section=section,
                        content_type=content_type,
                        content=content,
                        bbox=bbox_norm,
                        metadata={
                            "source": "paddleocr_ppstructure",
                            "block_type": btype,
                            "bbox_unit": "normalized",
                            "render_dpi": dpi,
                            "page_image_size": [w, h],
                        },
                    )
                    all_elements.append(el)

        return doc_id, doc.page_count, all_elements
