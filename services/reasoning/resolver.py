"""
resolver.py

Dynamically resolves company entities and sub-queries to specific document IDs
in Qdrant before executing evidence retrieval.

Uses:
1. Fast local catalog lookup (document_catalog.json) mapping company names/aliases -> document IDs.
2. Unsupervised entity discovery search via search_documents() as fallback for unknown entities.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from state import AgentState
from tools import search_documents

CATALOG_PATH = Path(__file__).resolve().parent / "document_catalog.json"

_DOCUMENT_CATALOG: Dict[str, Any] = {
    "doc_to_meta": {},
    "company_to_docs": {},
}

if CATALOG_PATH.exists():
    try:
        with open(CATALOG_PATH, "r", encoding="utf-8") as f:
            _DOCUMENT_CATALOG = json.load(f)
        print(f"[resolver] Loaded document catalog with {len(_DOCUMENT_CATALOG.get('company_to_docs', {}))} company keys.")
    except Exception as exc:
        print(f"[resolver] Warning: Failed to load document catalog: {exc}")

ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://orchestrator:8000")


def _persist_catalog():
    """Persists newly discovered company mappings to document_catalog.json."""
    try:
        with open(CATALOG_PATH, "w", encoding="utf-8") as f:
            json.dump(_DOCUMENT_CATALOG, f, indent=2)
        print(f"[resolver] Persisted updated catalog to {CATALOG_PATH}")
    except Exception as exc:
        print(f"[resolver] Note: Could not persist catalog: {exc}")


def sync_with_orchestrator():
    """
    Auto-syncs catalog with orchestrator's /documents registry.
    Automatically extracts company names from newly ingested filenames.
    """
    global _DOCUMENT_CATALOG
    try:
        import urllib.request
        url = f"{ORCHESTRATOR_URL.rstrip('/')}/documents"
        req = urllib.request.Request(url, headers={"User-Agent": "Ledger-Resolver"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            docs = json.loads(resp.read().decode())
            new_count = 0
            for d in docs:
                doc_id = d.get("document_id")
                filename = d.get("filename", "")
                if not doc_id:
                    continue
                # Extract clean company name from filename, e.g. "nvidia-corp_2023.pdf" -> "nvidia"
                base = filename.replace(".pdf", "").replace("_", "-")
                parts = base.split("-")
                if parts and parts[-1].isdigit():
                    parts = parts[:-1]
                comp_candidate = " ".join(parts).strip().lower()
                clean_comp = clean_entity_name(comp_candidate)
                if clean_comp:
                    comp_map = _DOCUMENT_CATALOG.setdefault("company_to_docs", {})
                    if clean_comp not in comp_map:
                        comp_map[clean_comp] = [doc_id]
                        new_count += 1
                    meta_map = _DOCUMENT_CATALOG.setdefault("doc_to_meta", {})
                    if doc_id not in meta_map:
                        meta_map[doc_id] = {
                            "company": comp_candidate.title(),
                            "source_document": filename,
                        }
            if new_count > 0:
                print(f"[resolver] Auto-synced {new_count} new documents from orchestrator.")
                _persist_catalog()
    except Exception:
        pass


# Run initial sync attempt on startup
sync_with_orchestrator()


def clean_entity_name(name: str) -> str:
    """Normalize company name for fuzzy matching."""
    if not name:
        return ""
    # Remove punctuation and common corporate suffixes
    s = re.sub(r"[^a-zA-Z0-9 ]+", " ", name).strip().lower()
    return " ".join(s.split())


def resolve_documents_for_entity(entity: str) -> List[str]:
    """
    Given an entity name (e.g., 'CTS', 'Jabil', 'Atlassian Corporation Plc'),
    returns a list of candidate document IDs.
    """
    cleaned = clean_entity_name(entity)
    if not cleaned:
        return []

    company_to_docs = _DOCUMENT_CATALOG.get("company_to_docs", {})

    # 1. Exact match in catalog
    if cleaned in company_to_docs:
        return company_to_docs[cleaned]

    # 2. Match without corporate suffixes
    stripped = re.sub(r"\b(inc|incorporated|corp|corporation|plc|ltd|limited|co|company|group)\b", "", cleaned).strip()
    stripped = " ".join(stripped.split())
    if stripped and stripped in company_to_docs:
        return company_to_docs[stripped]

    # 3. Substring matching in catalog
    for key, doc_ids in company_to_docs.items():
        if key == cleaned or key == stripped:
            return doc_ids
        if len(key) >= 3 and (key in cleaned or cleaned in key):
            return doc_ids
        if stripped and len(stripped) >= 3 and (key in stripped or stripped in key):
            return doc_ids

    # 4. Token-level match for distinctive names
    words = [w for w in stripped.split() if len(w) >= 3 and w not in ("the", "and", "for")]
    for w in words:
        if w in company_to_docs:
            return company_to_docs[w]

    # 5. Live sync with orchestrator in case new PDFs were just ingested
    sync_with_orchestrator()
    company_to_docs = _DOCUMENT_CATALOG.get("company_to_docs", {})
    if cleaned in company_to_docs:
        return company_to_docs[cleaned]

    # 6. Unsupervised Fallback: Entity search via retrieval API
    print(f"[resolver] Entity '{entity}' not found in catalog, executing discovery search...")
    try:
        discovery_chunks = search_documents(f"{entity} annual report financial statements", top_k=3)
        discovered_docs = list(dict.fromkeys(c.document_id for c in discovery_chunks if c.document_id))
        if discovered_docs:
            print(f"[resolver] Discovered candidate docs for '{entity}': {discovered_docs}")
            if "company_to_docs" in _DOCUMENT_CATALOG:
                _DOCUMENT_CATALOG["company_to_docs"][cleaned] = discovered_docs
                if stripped:
                    _DOCUMENT_CATALOG["company_to_docs"][stripped] = discovered_docs
                _persist_catalog()
            return discovered_docs
    except Exception as exc:
        print(f"[resolver] Discovery search failed for '{entity}': {exc}")

    return []


def resolve_documents(state: AgentState) -> AgentState:
    """
    LangGraph node: Inspects sub_queries and the question, resolves target
    document IDs for each sub-query, and annotates the state.
    """
    sub_queries = state.get("sub_queries", [])
    question = state.get("question", "")
    global_doc_id = state.get("document_id")

    resolved_map: Dict[str, List[str]] = {}
    updated_subqueries = []

    for sq in sub_queries:
        sq_dict = dict(sq) if isinstance(sq, dict) else sq.model_dump()
        entity = sq_dict.get("entity")

        # If no explicit entity, check if a known company appears in query or question
        if not entity:
            company_to_docs = _DOCUMENT_CATALOG.get("company_to_docs", {})
            q_text = (sq_dict.get("query", "") + " " + question).lower()
            for comp_key in company_to_docs:
                if len(comp_key) >= 3 and comp_key in q_text:
                    entity = comp_key
                    break

        candidate_docs = []
        if entity:
            candidate_docs = resolve_documents_for_entity(entity)
            resolved_map[entity] = candidate_docs

        if candidate_docs:
            sq_dict["candidate_document_ids"] = candidate_docs
            sq_dict["document_id"] = candidate_docs[0]
            print(f"[resolver] Subquery '{sq_dict.get('query')}' (entity '{entity}') -> resolved to {candidate_docs}")
        elif global_doc_id:
            sq_dict["candidate_document_ids"] = [global_doc_id]
            sq_dict["document_id"] = global_doc_id
        else:
            sq_dict["candidate_document_ids"] = []
            sq_dict["document_id"] = None

        updated_subqueries.append(sq_dict)

    state["sub_queries"] = updated_subqueries
    state["resolved_documents"] = resolved_map
    return state
