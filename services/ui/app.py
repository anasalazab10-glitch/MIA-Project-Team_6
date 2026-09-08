import os
from typing import Any, Dict, List, Tuple

import gradio as gr
import requests
from dotenv import load_dotenv

load_dotenv()

# Per the project README: orchestrator is reachable at localhost:8003 locally,
# or http://orchestrator:8000 inside the Docker network.
ORCHESTRATOR_URL = os.environ.get("ORCHESTRATOR_URL", "http://orchestrator:8000")


def call_backend(question: str) -> Dict[str, Any]:
    """
    Sends the question to the orchestrator. Falls back to a mock answer if
    the orchestrator isn't reachable yet (it doesn't exist as of writing this).
    """
    try:
        res = requests.post(
            f"{ORCHESTRATOR_URL}/run",
            json={"question": question},
            timeout=60,
        )
        if res.status_code == 200:
            return res.json()
        else:
            print(f"[call_backend] orchestrator returned status {res.status_code}: {res.text}")
    except Exception as exc:
        print(f"[call_backend] orchestrator call failed: {exc}")

    # Mock fallback - lets the UI be built/tested before orchestrator exists
    return {
        "answer_type": "insufficient_evidence",
        "evidence": [],
        "params": {
            "reason": "Orchestrator is not reachable yet - this is placeholder "
            "mock data so the UI can be built and tested independently."
        },
    }


def format_answer(answer: Dict[str, Any]) -> str:
    """Turns the strict answer JSON into readable Markdown for the chat view."""
    answer_type = answer.get("answer_type", "unknown")
    params = answer.get("params", {})
    evidence = answer.get("evidence", [])

    badge = {
        "direct": "🟢 Direct",
        "calculated": "🔵 Calculated",
        "multi_span": "🟣 Multi-item",
        "insufficient_evidence": "🔴 Insufficient Evidence",
    }.get(answer_type, f"⚪ {answer_type}")

    lines = [f"**{badge}**", ""]

    if answer_type == "direct":
        lines.append(f"**Answer:** {params.get('value', '—')}")
    elif answer_type == "calculated":
        lines.append(f"**Answer:** {params.get('value', '—')}")
        lines.append(f"**Formula:** `{params.get('formula', '—')}`")
    elif answer_type == "multi_span":
        values = params.get("values", [])
        lines.append("**Answer:** " + ", ".join(str(v) for v in values))
    else:  # insufficient_evidence
        lines.append(f"**Reason:** {params.get('reason', 'No further detail provided.')}")

    if evidence:
        lines.append("")
        lines.append("**Sources:**")
        for e in evidence:
            doc_id = e.get("document_id", "unknown")
            page = e.get("page", [])
            page_str = ", ".join(str(p) for p in page) if isinstance(page, list) else str(page)
            section = e.get("section", "")
            lines.append(f"- `{doc_id}`, page {page_str}" + (f" — {section}" if section else ""))
    else:
        lines.append("")
        lines.append("_No sources cited._")

    return "\n".join(lines)


def chat_respond(message: str, history: List[Dict[str, str]]) -> Tuple[str, List[Dict[str, str]]]:
    answer = call_backend(message)
    formatted = format_answer(answer)

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": formatted},
    ]
    return "", history


def fetch_documents() -> List[List[Any]]:
    try:
        res = requests.get(f"{ORCHESTRATOR_URL}/documents", timeout=60)
        if res.status_code == 200:
            docs = res.json()
            if not docs:
                return [["No documents indexed yet.", "", "", "", ""]]
            return [
                [
                    d.get("document_id", ""),
                    d.get("filename", ""),
                    d.get("num_pages", 0),
                    d.get("num_tables", 0),
                    d.get("created_at", ""),
                ]
                for d in docs
            ]
        else:
            print(f"[fetch_documents] orchestrator returned status {res.status_code}: {res.text}")
    except Exception as exc:
        print(f"[fetch_documents] orchestrator call failed: {exc}")

    return [["Orchestrator not reachable yet", "-", "-", "-", "-"]]


def fetch_dashboard() -> Tuple[str, List[List[Any]]]:
    try:
        res = requests.get(f"{ORCHESTRATOR_URL}/dashboard/stats", timeout=60)
        if res.status_code == 200:
            stats = res.json()
            summary = (
                f"**Total documents:** {stats.get('total_documents', 0)}  \n"
                f"**Total tables:** {stats.get('total_tables', 0)}  \n"
                f"**Total elements:** {stats.get('total_elements', 0)}"
            )
            recent = stats.get("recent_queries", [])
            rows = [
                [
                    q.get("question", "")[:60],
                    q.get("answer_type", ""),
                    "✅" if q.get("validation_passed") else "❌",
                    f"{q.get('latency_ms', 0):.0f} ms",
                    q.get("timestamp", ""),
                ]
                for q in reversed(recent)
            ]
            return summary, rows
        else:
            print(f"[fetch_dashboard] orchestrator returned status {res.status_code}: {res.text}")
    except Exception as exc:
        print(f"[fetch_dashboard] orchestrator call failed: {exc}")

    return (
        "_Orchestrator not reachable yet — showing no data._",
        [["-", "-", "-", "-", "-"]],
    )


with gr.Blocks(title="LEDGER - Financial Document Intelligence Agent") as demo:
    gr.Markdown("# LEDGER — Financial Document Intelligence Agent")

    with gr.Tab("Chat"):
        chatbot = gr.Chatbot(height=500)
        msg = gr.Textbox(
            placeholder="Ask a question about the indexed financial reports...",
            label="Your question",
        )
        clear = gr.Button("Clear chat")

        msg.submit(chat_respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
        clear.click(lambda: [], outputs=chatbot)

    with gr.Tab("Documents"):
        gr.Markdown("### Indexed Documents")
        doc_table = gr.Dataframe(
            headers=["Document ID", "Filename", "Pages", "Tables", "Created At"],
            value=fetch_documents(),
            interactive=False,
        )
        refresh_docs_btn = gr.Button("Refresh")
        refresh_docs_btn.click(fetch_documents, outputs=doc_table)

    with gr.Tab("Dashboard"):
        gr.Markdown("### Corpus Overview")
        summary_md = gr.Markdown(fetch_dashboard()[0])
        gr.Markdown("### Recent Queries")
        queries_table = gr.Dataframe(
            headers=["Question", "Answer Type", "Validated", "Latency", "Timestamp"],
            value=fetch_dashboard()[1],
            interactive=False,
        )
        refresh_dashboard_btn = gr.Button("Refresh")
        refresh_dashboard_btn.click(fetch_dashboard, outputs=[summary_md, queries_table])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)