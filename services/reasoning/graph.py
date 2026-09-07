
from langgraph.graph import StateGraph, START, END

from state import AgentState
from classifier import classify_question
from retriever import retrieve_evidence, should_retry, MAX_RETRIES
from reasoner import reason_over_evidence
from formatter import format_answer


def mark_insufficient(state: AgentState) -> AgentState:
    state["question_type"] = "insufficient_evidence"
    if not state.get("reasoning_summary"):
        state["reasoning_summary"] = (
            "Could not find sufficient evidence to answer this question "
            f"after {state.get('retry_count', 0)} retrieval attempt(s)."
        )
    return state


def route_after_retrieval(state: AgentState) -> str:
    """
    Conditional edge: the core 'sufficient vs. insufficient, retry on weak
    evidence' branch required by the spec.
    """
    status = state.get("evidence_status")

    if status == "sufficient":
        return "reasoner"

    if should_retry(state):
        return "retriever"  # loop back for another search attempt

    return "mark_insufficient"  # out of retries, give up honestly


def build_graph():
    workflow = StateGraph(AgentState)

    # Register nodes
    workflow.add_node("classifier", classify_question)
    workflow.add_node("retriever", retrieve_evidence)
    workflow.add_node("reasoner", reason_over_evidence)
    workflow.add_node("formatter", format_answer)
    workflow.add_node("mark_insufficient", mark_insufficient)

    # Entry point
    workflow.add_edge(START, "classifier")
    workflow.add_edge("classifier", "retriever")

    # The real conditional branch: sufficient -> reasoner,
    # weak/insufficient + retries left -> retry, else -> give up
    workflow.add_conditional_edges(
        "retriever",
        route_after_retrieval,
        {
            "reasoner": "reasoner",
            "retriever": "retriever",
            "mark_insufficient": "mark_insufficient",
        },
    )

    workflow.add_edge("reasoner", "formatter")
    workflow.add_edge("mark_insufficient", "formatter")
    workflow.add_edge("formatter", END)

    return workflow.compile()


# Compiled graph, ready to be imported by main.py and run_test.py
agent_graph = build_graph()


# Standalone test block
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    test_questions = [
        "What was Jabil Circuit's operating income in 2019?"
    ]

    for q in test_questions:
        print(f"\n{'='*70}")
        print(f"QUESTION: {q}")
        print("-" * 70)

        initial_state = {
            "question": q,
            "session_id": "test-session",
            "retrieved_chunks": [],
        }

        result = agent_graph.invoke(initial_state)

        print("question_type   :", result.get("question_type"))
        print("evidence_status :", result.get("evidence_status"))
        print("retry_count     :", result.get("retry_count"))
        print("final_answer    :", result.get("final_answer"))
