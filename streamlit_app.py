"""
Streamlit chat + dashboard UI for the RAG pipeline.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

import streamlit as st

from src.pipeline import RAGPipeline


@st.cache_resource(show_spinner=False)
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


def _format_answer(result: Dict[str, Any]) -> str:
    return str(result.get("final_answer", "")).strip() or "I could not generate a reliable answer."


def _metrics_row(query: str, result: Dict[str, Any], elapsed_ms: float) -> Dict[str, Any]:
    return {
        "query": query,
        "complexity": result.get("complexity", ""),
        "model_final": result.get("model_used_final", ""),
        "confidence_final": result.get("confidence_score_final", ""),
        "retried": result.get("retried", False),
        "retrieval_retry": result.get("retrieval_retry", False),
        "cache_hit": result.get("cache_hit", False),
        "coverage_score": result.get("coverage_score", ""),
        "total_pipeline_ms": result.get("total_pipeline_ms", ""),
        "ui_elapsed_ms": round(elapsed_ms, 2),
        "input_tokens": result.get("input_tokens", ""),
        "output_tokens": result.get("output_tokens", ""),
    }


def _ensure_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "runs" not in st.session_state:
        st.session_state.runs = []


def _inject_styles() -> None:
    st.markdown(
        """
<style>
.block-container {padding-top: 0.9rem; max-width: 1180px;}
.app-card {
  background: #141619;
  border: 1px solid #2a2d34;
  border-radius: 18px;
  padding: 14px 16px;
  margin-bottom: 10px;
}
.brand-title {font-size: 1.7rem; font-weight: 700; color: #f3f4f6; margin-bottom: 2px;}
.brand-sub {color: #a3a9b5; font-size: 0.95rem;}
.answer-card {
  background: #1c1f25;
  border: 1px solid #2d323b;
  border-radius: 14px;
  padding: 13px 15px;
  color: #f4f6fb;
  font-size: 1.08rem;
  line-height: 1.45;
  margin-top: 4px;
}
.chip-row {display: flex; gap: 8px; flex-wrap: wrap; margin-top: 9px;}
.chip {
  background: #1e242d;
  border: 1px solid #364152;
  color: #e0e5ef;
  border-radius: 999px;
  padding: 5px 11px;
  font-size: 0.88rem;
  font-weight: 500;
}
.metric-grid {display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;}
.metric-card {
  background: #1c1f25;
  border: 1px solid #2d323b;
  border-radius: 12px;
  padding: 12px;
}
.metric-k {color: #f5f7fb; font-size: 1.7rem; font-weight: 700; line-height: 1.1;}
.metric-l {color: #a8afbc; font-size: 0.86rem; margin-top: 2px;}
.section-h {color: #d6dce7; font-size: 0.9rem; letter-spacing: 0.03em; margin: 8px 0 8px 0; font-weight: 600;}
.route-grid {display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;}
.route-card {
  background: #1c1f25;
  border: 1px solid #2d323b;
  border-radius: 12px;
  padding: 12px;
  text-align: center;
}
.route-k {color: #f5f7fb; font-size: 2rem; font-weight: 700; line-height: 1;}
.route-l {color: #a8afbc; font-size: 0.88rem; margin-top: 4px;}
</style>
        """,
        unsafe_allow_html=True,
    )


def _render_header() -> None:
    st.markdown(
        """
<div class="app-card">
  <div class="brand-title">Notre Dame Assistant</div>
  <div class="brand-sub">University of Notre Dame · cost-optimized RAG · GPU2_Group1</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def _render_dashboard() -> None:
    st.markdown('<div class="section-h">SESSION SUMMARY</div>', unsafe_allow_html=True)
    runs: List[Dict[str, Any]] = st.session_state.runs
    if not runs:
        st.info("No runs yet. Ask a question to populate metrics.")
        return

    latest = runs[-1]
    avg_latency = round(
        sum(float(r.get("total_pipeline_ms", 0) or 0) for r in runs) / max(len(runs), 1),
        2,
    )
    avg_conf = round(
        sum(float(r.get("confidence_final", 0) or 0) for r in runs) / max(len(runs), 1),
        2,
    )
    total_saved = int(
        sum(
            max(
                0,
                int(float(r.get("input_tokens", 0) or 0) - float(r.get("output_tokens", 0) or 0)),
            )
            for r in runs
        )
    )
    total_in = int(sum(float(r.get("input_tokens", 0) or 0) for r in runs))
    total_out = int(sum(float(r.get("output_tokens", 0) or 0) for r in runs))
    local_ratio = (
        sum(1 for r in runs if "gpt-" not in str(r.get("model_final", "")).lower()) / max(len(runs), 1)
    ) * 100

    # Lightweight trend charts for presentation.
    latency_vals = [float(r.get("total_pipeline_ms", 0) or 0) for r in runs]
    confidence_vals = [float(r.get("confidence_final", 0) or 0) for r in runs]
    input_token_vals = [float(r.get("input_tokens", 0) or 0) for r in runs]
    output_token_vals = [float(r.get("output_tokens", 0) or 0) for r in runs]
    complexity_counts: Dict[str, int] = {}
    for r in runs:
        key = str(r.get("complexity", "") or "unknown")
        complexity_counts[key] = complexity_counts.get(key, 0) + 1

    st.markdown(
        f"""
<div class="metric-grid">
  <div class="metric-card"><div class="metric-k">{avg_latency:.1f} ms</div><div class="metric-l">⏱ avg latency</div></div>
  <div class="metric-card"><div class="metric-k">{avg_conf:.2f}</div><div class="metric-l">🎯 avg confidence</div></div>
  <div class="metric-card"><div class="metric-k">{total_saved}</div><div class="metric-l">💾 tokens saved</div></div>
  <div class="metric-card"><div class="metric-k">{total_in}</div><div class="metric-l">📥 prompt tokens</div></div>
  <div class="metric-card"><div class="metric-k">{total_out}</div><div class="metric-l">📤 completion tokens</div></div>
  <div class="metric-card"><div class="metric-k">{local_ratio:.0f}%</div><div class="metric-l">🧠 local query ratio</div></div>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-h">MODEL ROUTING</div>', unsafe_allow_html=True)
    simple_count = complexity_counts.get("simple", 0)
    medium_count = complexity_counts.get("medium", 0)
    complex_count = complexity_counts.get("complex", 0)
    st.markdown(
        f"""
<div class="route-grid">
  <div class="route-card"><div class="route-k">{simple_count}</div><div class="route-l">🟢 simple</div></div>
  <div class="route-card"><div class="route-k">{medium_count}</div><div class="route-l">🟡 medium</div></div>
  <div class="route-card"><div class="route-k">{complex_count}</div><div class="route-l">🔴 complex</div></div>
</div>
        """,
        unsafe_allow_html=True,
    )

    st.caption("Latency Trend (ms)")
    st.line_chart({"latency_ms": latency_vals})

    st.caption("Confidence Trend")
    st.line_chart({"confidence": confidence_vals})

    st.caption("Token Usage Trend (Input vs Output)")
    st.line_chart(
        {
            "input_tokens": input_token_vals,
            "output_tokens": output_token_vals,
        }
    )

    st.caption("Complexity Distribution")
    st.bar_chart(complexity_counts)

    st.caption("Recent Queries")
    st.dataframe(list(reversed(runs[-10:])), use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="RAG Chat + Dashboard", layout="wide")
    _inject_styles()
    _render_header()

    _ensure_state()
    pipeline = get_pipeline()

    tab_chat, tab_dashboard = st.tabs(["Chat", "Dashboard"])

    with tab_chat:
        st.subheader("Chat")
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        prompt = st.chat_input("Ask about the current course material...")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Running pipeline..."):
                    t0 = time.perf_counter()
                    result = pipeline.run(prompt)
                    elapsed_ms = (time.perf_counter() - t0) * 1000

                answer = _format_answer(result)
                st.markdown(f'<div class="answer-card">{answer}</div>', unsafe_allow_html=True)

                q_type = str(result.get("complexity", ""))
                model_used = str(result.get("model_used_final", ""))
                conf = result.get("confidence_score_final", "")
                latency = result.get("total_pipeline_ms", "")
                saved = int(float(result.get("input_tokens", 0) or 0) - float(result.get("output_tokens", 0) or 0))
                k_val = result.get("k", "")
                st.markdown(
                    (
                        f'<div class="chip-row">'
                        f'<span class="chip">🧭 {q_type}</span>'
                        f'<span class="chip">🤖 {model_used}</span>'
                        f'<span class="chip">⏱ {latency} ms</span>'
                        f'<span class="chip">🎯 conf {conf}</span>'
                        f'<span class="chip">💾 {saved} saved</span>'
                        f'<span class="chip">📚 k={k_val}</span>'
                        f'</div>'
                    ),
                    unsafe_allow_html=True,
                )

            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.session_state.runs.append(_metrics_row(prompt, result, elapsed_ms))
            st.rerun()

    with tab_dashboard:
        _render_dashboard()


if __name__ == "__main__":
    main()
