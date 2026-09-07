"""
Metadata and audit catalog for the LEDGER Orchestrator Service.
Provides in-memory fast access and lightweight JSON file persistence
for indexed documents, extracted tables, and query audit history.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from schemas import (
    DashboardStats,
    DocumentDetail,
    DocumentSummary,
    QueryAuditItem,
    TableSummary,
)


class MetadataStore:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.docs_file = self.data_dir / "documents_catalog.json"
        self.queries_file = self.data_dir / "query_history.json"

        self.documents: Dict[str, DocumentDetail] = {}
        self.queries: List[QueryAuditItem] = []

        self._load()

    def _load(self) -> None:
        if self.docs_file.exists():
            try:
                with open(self.docs_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        doc = DocumentDetail.model_validate(item)
                        self.documents[doc.document_id] = doc
            except Exception as exc:
                print(f"[MetadataStore] Warning: Could not load {self.docs_file}: {exc}")

        if self.queries_file.exists():
            try:
                with open(self.queries_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.queries = [QueryAuditItem.model_validate(q) for q in data]
            except Exception as exc:
                print(f"[MetadataStore] Warning: Could not load {self.queries_file}: {exc}")

    def _save_docs(self) -> None:
        try:
            with open(self.docs_file, "w", encoding="utf-8") as f:
                json.dump(
                    [doc.model_dump() for doc in self.documents.values()],
                    f,
                    indent=2,
                )
        except Exception as exc:
            print(f"[MetadataStore] Error saving documents catalog: {exc}")

    def _save_queries(self) -> None:
        try:
            with open(self.queries_file, "w", encoding="utf-8") as f:
                json.dump(
                    [q.model_dump() for q in self.queries[-100:]],
                    f,
                    indent=2,
                )
        except Exception as exc:
            print(f"[MetadataStore] Error saving query history: {exc}")

    def record_document(
        self,
        document_id: str,
        filename: str,
        num_pages: int,
        elements: List[Dict[str, Any]],
    ) -> DocumentDetail:
        tables: List[TableSummary] = []
        for el in elements:
            if el.get("content_type") == "table":
                content = el.get("content", {})
                headers = []
                rows = []
                if isinstance(content, dict):
                    headers = content.get("headers", [])
                    rows = content.get("rows", [])
                elif isinstance(content, str):
                    headers = ["Extracted Table Content"]
                    rows = [[content[:100]]]

                preview = rows[:3] if rows else []
                tables.append(
                    TableSummary(
                        document_id=document_id,
                        page=el.get("page", 1),
                        section=el.get("section"),
                        headers=headers,
                        num_rows=len(rows),
                        preview=preview,
                    )
                )

        elements_preview = [
            {
                "chunk_id": el.get("chunk_id", ""),
                "page": el.get("page", 1),
                "content_type": el.get("content_type", "text"),
                "section": el.get("section"),
                "preview": str(el.get("content", ""))[:120],
            }
            for el in elements[:10]
        ]

        doc = DocumentDetail(
            document_id=document_id,
            filename=filename,
            num_pages=num_pages,
            num_elements=len(elements),
            num_tables=len(tables),
            created_at=datetime.now(timezone.utc).isoformat(),
            tables=tables,
            elements_preview=elements_preview,
        )

        self.documents[document_id] = doc
        self._save_docs()
        return doc

    def get_document(self, document_id: str) -> Optional[DocumentDetail]:
        return self.documents.get(document_id)

    def list_documents(self) -> List[DocumentSummary]:
        return [
            DocumentSummary(
                document_id=doc.document_id,
                filename=doc.filename,
                num_pages=doc.num_pages,
                num_elements=doc.num_elements,
                num_tables=doc.num_tables,
                created_at=doc.created_at,
            )
            for doc in self.documents.values()
        ]

    def record_query(
        self,
        question: str,
        answer_type: str,
        validation_passed: bool,
        latency_ms: float,
        evidence_count: int = 0,
    ) -> QueryAuditItem:
        item = QueryAuditItem(
            query_id=str(uuid.uuid4())[:8],
            question=question,
            answer_type=answer_type,
            validation_passed=validation_passed,
            latency_ms=round(latency_ms, 2),
            timestamp=datetime.now(timezone.utc).isoformat(),
            evidence_count=evidence_count,
        )
        self.queries.append(item)
        # Keep last 100 queries
        if len(self.queries) > 100:
            self.queries = self.queries[-100:]
        self._save_queries()
        return item

    def get_dashboard_stats(self) -> DashboardStats:
        all_tables: List[TableSummary] = []
        total_elements = 0
        for doc in self.documents.values():
            all_tables.extend(doc.tables)
            total_elements += doc.num_elements

        return DashboardStats(
            total_documents=len(self.documents),
            total_tables=len(all_tables),
            total_elements=total_elements,
            documents=self.list_documents(),
            tables=all_tables[:20],
            recent_queries=list(reversed(self.queries[-20:])),
        )
