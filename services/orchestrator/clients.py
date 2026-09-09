"""
Downstream HTTP service clients for the LEDGER Orchestrator.
Communicates asynchronously with Doc Processor, Retrieval, Reasoning, and Validator services.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import httpx

from config import settings

logger = logging.getLogger("orchestrator.clients")


class ServiceClients:
    def __init__(self):
        self._http_client: Optional[httpx.AsyncClient] = None

    async def start(self) -> None:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=60.0)

    async def close(self) -> None:
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()

    @property
    def client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=60.0)
        return self._http_client

    # ------------------------------------------------------------------
    # 1. Document Processor Client
    # ------------------------------------------------------------------
    async def process_document(
        self,
        pdf_bytes: bytes,
        filename: str,
        document_id: Optional[str] = None,
        dpi: int = 200,
    ) -> Dict[str, Any]:
        """
        Sends raw PDF bytes to doc-processor-api (/process).
        Returns ProcessResponse dict: { document_id, num_pages, elements }
        """
        url = f"{settings.doc_processor_url}/process"
        files = {"file": (filename, pdf_bytes, "application/pdf")}
        data = {"dpi": str(dpi)}
        if document_id:
            data["document_id"] = document_id

        try:
            res = await self.client.post(
                url,
                files=files,
                data=data,
                timeout=settings.doc_processor_timeout,
            )
            res.raise_for_status()
            return res.json()
        except httpx.HTTPStatusError as exc:
            logger.error(f"[DocProcessor] HTTP error {exc.response.status_code}: {exc.response.text}")
            raise RuntimeError(f"DocProcessor returned {exc.response.status_code}: {exc.response.text}") from exc
        except Exception as exc:
            logger.error(f"[DocProcessor] Call failed at {url}: {exc}")
            raise RuntimeError(f"Failed to communicate with DocProcessor at {url}: {exc}") from exc

    # ------------------------------------------------------------------
    # 2. Retrieval Client
    # ------------------------------------------------------------------
    async def search_retrieval(
        self,
        query: str,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Queries retrieval-api (/search).
        """
        url = f"{settings.retrieval_url}/search"
        payload = {"query": query}
        if metadata_filter:
            payload["metadata_filter"] = metadata_filter

        try:
            res = await self.client.post(
                url,
                json=payload,
                timeout=settings.retrieval_timeout,
            )
            res.raise_for_status()
            return res.json()
        except Exception as exc:
            logger.warning(f"[Retrieval] Search call failed at {url}: {exc}")
            return {"results": [], "total": 0, "error": str(exc)}

    async def index_elements(
        self,
        document_id: str,
        elements: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Sends processed elements to retrieval-api for chunking & indexing.
        If retrieval-api exposes /index, it forwards them. Otherwise, logs a note.
        """
        url = f"{settings.retrieval_url}/index"
        payload = {"document_id": document_id, "elements": elements}

        try:
            res = await self.client.post(
                url,
                json=payload,
                timeout=settings.retrieval_timeout,
            )
            if res.status_code == 200:
                return res.json()
            elif res.status_code == 404:
                logger.info(f"[Retrieval] /index endpoint not implemented yet on retrieval service.")
                return {"status": "pending_retrieval_indexing", "count": len(elements)}
            else:
                logger.warning(f"[Retrieval] /index returned {res.status_code}: {res.text}")
                return {"status": "retrieval_error", "detail": res.text}
        except Exception as exc:
            logger.info(f"[Retrieval] /index not reachable: {exc}")
            return {"status": "retrieval_offline", "detail": str(exc)}

    # ------------------------------------------------------------------
    # 3. Reasoning Agent Client
    # ------------------------------------------------------------------
    async def run_reasoning(
        self,
        question: str,
        session_id: Optional[str] = None,
        document_id: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Sends the question to reasoning agent-service (/run).
        Returns raw answer JSON from LangGraph.
        """
        url = f"{settings.reasoning_url}/run"
        payload: Dict[str, Any] = {"question": question}
        if session_id:
            payload["session_id"] = session_id
        if document_id:
            payload["document_id"] = document_id
        if trace_id:
            payload["trace_id"] = trace_id

        try:
            res = await self.client.post(
                url,
                json=payload,
                timeout=settings.reasoning_timeout,
            )
            res.raise_for_status()
            return res.json()
        except httpx.HTTPStatusError as exc:
            logger.error(f"[Reasoning] HTTP error {exc.response.status_code}: {exc.response.text}")
            raise RuntimeError(f"Reasoning service returned {exc.response.status_code}: {exc.response.text}") from exc
        except Exception as exc:
            logger.error(f"[Reasoning] Call failed at {url}: {exc}")
            raise RuntimeError(f"Reasoning service unreachable at {url}: {exc}") from exc

    # ------------------------------------------------------------------
    # 4. Answer Validator Client
    # ------------------------------------------------------------------
    
    async def validate_answer(
        self,
        answer_payload: Dict[str, Any],
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Sends the candidate answer to validator-service (/validate_answer).
        Returns { valid: bool, reason: str }
        """
        url = f"{settings.validator_url}/validate_answer"
    
        payload = dict(answer_payload)
    
        if trace_id:
            payload["trace_id"] = trace_id
    
        try:
            res = await self.client.post(
                url,
                json=payload,
                timeout=settings.validator_timeout,
            )
    
            if res.status_code == 200:
                return res.json()
            else:
                logger.error(
                    f"[Validator] Unexpected status "
                    f"{res.status_code}: {res.text}"
                )
                return {
                    "valid": False,
                    "reason": f"Validator HTTP {res.status_code}: {res.text}",
                }
    
        except Exception as exc:
            logger.error(f"[Validator] Call failed at {url}: {exc}")
    
            # If validator is unreachable, we must not let unvalidated answer pass!
            return {
                "valid": False,
                "reason": f"Validator service unreachable at {url}: {exc}",
            }



    # ------------------------------------------------------------------
    # Health checks
    # ------------------------------------------------------------------
    async def check_health(self) -> Dict[str, Dict[str, Any]]:
        services = {
            "doc-processor": f"{settings.doc_processor_url}/health",
            "retrieval": f"{settings.retrieval_url}/health",
            "reasoning": f"{settings.reasoning_url}/health",
            "validator": f"{settings.validator_url}/health",
        }
        results = {}
        for name, url in services.items():
            try:
                res = await self.client.get(url, timeout=3.0)
                results[name] = {
                    "reachable": res.status_code == 200,
                    "status_code": res.status_code,
                    "url": url,
                }
            except Exception as exc:
                results[name] = {
                    "reachable": False,
                    "error": str(exc),
                    "url": url,
                }
        return results
