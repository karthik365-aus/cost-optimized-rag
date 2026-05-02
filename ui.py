"""
ui.py — Streamlit chat UI for the Cost-Optimized RAG pipeline.

Run:
    streamlit run ui.py

Fixes applied v2:
  - Single-word queries no longer blocked as off-topic
  - Conversation continuity: short follow-ups inherit prior context
  - HTML rendering leak fixed (off-topic uses st.warning, not custom HTML)
  - Pricing dashboard: actual vs baseline cost, $ saved, % reduction
"""

import io
from pathlib import Path
import re
import time
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Assistant",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=DM+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }

.nd-header {
    display: flex; align-items: center; gap: 14px;
    padding: 18px 0 14px;
    border-bottom: 1px solid rgba(0,0,0,0.08);
    margin-bottom: 20px;
}
.nd-logo {
    width: 54px; height: 54px; background: #2563EB;
    border-radius: 10px; display: flex; align-items: center;
    justify-content: center; font-size: 30px; flex-shrink: 0;
    color: #FFFFFF;
    box-shadow: 0 2px 8px rgba(37, 99, 235, 0.35);
}
.nd-title { font-size: 24px; font-weight: 800; color: #1D4ED8; margin: 0; }
.nd-sub   { font-size: 12px; color: #666; margin: 2px 0 0; }

.chat-wrap { display: flex; flex-direction: column; gap: 16px; padding: 8px 0; }

.msg-user { display: flex; justify-content: flex-end; gap: 10px; align-items: flex-start; }
.msg-bot  { display: flex; justify-content: flex-start; gap: 10px; align-items: flex-start; }

.avatar {
    width: 32px; height: 32px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 12px; font-weight: 600; flex-shrink: 0;
}
.avatar-bot  { background: #0C2340; color: #C99700; }
.avatar-user { background: #F0F0F0; color: #444; border: 1px solid #ddd; }

.bubble {
    max-width: 78%; padding: 11px 15px;
    border-radius: 14px; font-size: 14px; line-height: 1.55;
}
.bubble-bot {
    background: #F7F7F7; color: #1a1a1a;
    border: 1px solid #e8e8e8; border-top-left-radius: 4px;
}
.bubble-user {
    background: #0C2340; color: #fff; border-top-right-radius: 4px;
}

.meta-strip { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 8px; }
.pill {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 3px 9px; border-radius: 20px;
    font-size: 11px; font-weight: 500;
    border: 1px solid #e0e0e0; background: #fafafa; color: #555;
    font-family: 'DM Mono', monospace;
}
.dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.dot-simple  { background: #3B6D11; }
.dot-medium  { background: #BA7517; }
.dot-complex { background: #A32D2D; }
.dot-cache   { background: #15803D; }

/* ── Dashboard stat cards ── */
.stat-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 10px; margin-bottom: 20px; }
.stat-card { background: #F7F8FA; border-radius: 10px; padding: 14px 16px; border: 1px solid #EAECEF; }
.stat-label { font-size: 11px; color: #888; margin-bottom: 4px; }
.stat-val   { font-size: 22px; font-weight: 600; color: #0C2340; }
.stat-sub   { font-size: 10px; color: #aaa; margin-top: 2px; }

/* ── Pricing cards ── */
.price-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 10px; margin-bottom: 20px; }
.price-card { border-radius: 10px; padding: 14px 16px; border: 1px solid #EAECEF; }
.price-card-spend    { background: #F7F8FA; }
.price-card-baseline { background: #FFF8F0; border-color: #FDDCB0; }
.price-card-saved    { background: #F0FAF4; border-color: #A7DFB8; }
.price-card-pct      { background: #0C2340; border-color: #0C2340; }
.price-label         { font-size: 11px; color: #888; margin-bottom: 4px; }
.price-label-inv     { font-size: 11px; color: #8EAAC8; margin-bottom: 4px; }
.price-val           { font-size: 22px; font-weight: 600; color: #0C2340; }
.price-val-saved     { font-size: 22px; font-weight: 600; color: #15803D; }
.price-val-inv       { font-size: 22px; font-weight: 600; color: #C99700; }
.price-sub           { font-size: 10px; color: #aaa; margin-top: 2px; }
.price-sub-inv       { font-size: 10px; color: #8EAAC8; margin-top: 2px; }

.price-breakdown {
    background: #F7F8FA; border-radius: 10px;
    padding: 14px 18px; border: 1px solid #EAECEF;
    margin-bottom: 20px; font-size: 12px; color: #555;
}
.price-row { display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #F0F0F0; }
.price-row:last-child { border-bottom: none; font-weight: 600; color: #0C2340; }
.price-row-green { color: #15803D; }

/* ── Routing cards ── */
.route-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 10px; margin-bottom: 20px; }
.route-card { background: #F7F8FA; border-radius: 10px; padding: 12px; text-align: center; border: 1px solid #EAECEF; }
.route-count { font-size: 26px; font-weight: 600; color: #0C2340; }
.route-label { font-size: 11px; color: #666; margin-top: 3px; }
.route-model { font-size: 10px; color: #aaa; margin-top: 2px; font-family: 'DM Mono', monospace; }

/* ── Query table ── */
.q-table { width: 100%; border-collapse: collapse; font-size: 12.5px; table-layout: fixed; }
.q-table th {
    text-align: left; padding: 7px 10px;
    border-bottom: 1px solid #EAECEF; color: #888;
    font-weight: 500; font-size: 11px;
}
.q-table td {
    padding: 8px 10px; border-bottom: 1px solid #F0F0F0;
    color: #333; vertical-align: middle;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.q-table tr:last-child td { border-bottom: none; }

.badge { display: inline-flex; align-items: center; padding: 2px 8px; border-radius: 20px; font-size: 10.5px; font-weight: 600; }
.badge-simple  { background: #EAF3DE; color: #3B6D11; }
.badge-medium  { background: #FAEEDA; color: #854F0B; }
.badge-complex { background: #FCEBEB; color: #A32D2D; }

.model-chip {
    display: inline-block; padding: 2px 7px; border-radius: 5px;
    font-size: 10px; background: #F0F0F0; border: 1px solid #ddd;
    color: #555; font-family: 'DM Mono', monospace;
}
.conf-bar-wrap { display: flex; align-items: center; gap: 6px; }
.conf-bar-bg   { flex: 1; height: 5px; background: #E8E8E8; border-radius: 3px; overflow: hidden; }
.conf-bar-fill { height: 100%; border-radius: 3px; background: #0C2340; }

.section-title {
    font-size: 11px; font-weight: 600; color: #888;
    text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)


# ── Pricing constants (gpt-4o-mini rates, April 2025) ─────────────────────────
GPT4O_MINI_INPUT_PER_M  = 0.150   # $ per million input tokens
GPT4O_MINI_OUTPUT_PER_M = 0.600   # $ per million output tokens


def calc_cost(input_tok: int, output_tok: int, is_local: bool) -> float:
    """Actual cost: local = free, OpenAI = gpt-4o-mini rates."""
    if is_local:
        return 0.0
    return (input_tok * GPT4O_MINI_INPUT_PER_M + output_tok * GPT4O_MINI_OUTPUT_PER_M) / 1_000_000


def calc_baseline_cost(original_tok: int, output_tok: int) -> float:
    """Naive baseline: every query goes to gpt-4o-mini at full uncompressed tokens."""
    return (original_tok * GPT4O_MINI_INPUT_PER_M + output_tok * GPT4O_MINI_OUTPUT_PER_M) / 1_000_000


def fmt_cost(dollars: float) -> str:
    if dollars == 0:
        return "$0.0000"
    if dollars < 0.001:
        return f"${dollars:.6f}"
    return f"${dollars:.4f}"


def is_openai_billed_result(result: dict) -> bool:
    """Infer paid path from router source or final model name."""
    router_source = (result.get("router_model_source") or "").strip().lower()
    model_final = (result.get("model_used_final") or "").strip().lower()
    if router_source == "openai":
        return True
    return model_final.startswith("gpt-") or "gpt-" in model_final


# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("messages", []),
    ("query_log", []),
    ("pipeline", None),
    ("pipeline_error", None),
    ("active_session_title", "AI Assistant"),
    ("session_retrieval_mode", "upload_only"),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Pipeline loader ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline():
    try:
        from src.pipeline import RAGPipeline
        return RAGPipeline(), None
    except Exception as e:
        return None, str(e)


def _extract_session_upload(uploaded_file):
    """Best-effort extraction + metadata for session uploads (PDF/TXT)."""
    name = (uploaded_file.name or "").lower()
    data = uploaded_file.getvalue()
    payload = {
        "name": uploaded_file.name or "",
        "mime_type": getattr(uploaded_file, "type", "") or "",
        "bytes": None,
        "pdf_text_by_page": {},
        "text": "",
    }
    if not data:
        return payload
    if name.endswith(".txt"):
        try:
            payload["text"] = data.decode("utf-8", errors="ignore").strip()
            return payload
        except Exception:
            return payload
    if name.endswith(".pdf"):
        payload["bytes"] = data
        try:
            import fitz

            doc = fitz.open(stream=data, filetype="pdf")
            pages = []
            text_by_page = {}
            try:
                for page_num, page in enumerate(doc):
                    text = (page.get_text("text", sort=True) or "").strip()
                    if text:
                        pages.append(text)
                        text_by_page[page_num] = text
            finally:
                doc.close()
            payload["pdf_text_by_page"] = text_by_page
            payload["text"] = "\n\n".join(pages).strip()
        except Exception:
            try:
                from pypdf import PdfReader

                reader = PdfReader(io.BytesIO(data))
                pages = []
                text_by_page = {}
                for page_num, page in enumerate(reader.pages):
                    text = (page.extract_text() or "").strip()
                    if text:
                        pages.append(text)
                        text_by_page[page_num] = text
                payload["pdf_text_by_page"] = text_by_page
                payload["text"] = "\n\n".join(pages).strip()
            except Exception:
                return payload
        return payload
    return payload


def _render_header(title: str):
    safe_title = (title or "AI Assistant").strip()
    st.markdown(
        f"""
<div class="nd-header">
  <div class="nd-logo">🤖</div>
  <div>
    <div class="nd-title">{safe_title}</div>
    <div class="nd-sub">Cost-Optimized RAG Assistant</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


# ── Off-topic guard ───────────────────────────────────────────────────────────
CHITCHAT = {
    "hi", "hello", "hey", "thanks", "thank you", "bye", "goodbye",
    "ok", "okay", "cool", "great", "nice", "sure", "yes", "no",
    "yep", "nope", "lol", "haha", "hmm",
}
PERSONAL_PATTERNS = [
    r"^my name is",
    r"^i am\b",
    r"^i'm\b",
    r"^i like\b",
    r"^i want\b",
    r"^i need\b",
    r"^tell me a joke",
    r"^who are you",
    r"^what are you",
    r"^what('s| is) the weather",
    r"^what('s| is) the time",
    r"^can you help me with",
    r"^what do you think about",
]


def is_off_topic(query: str) -> bool:
    """Layer 1 — pattern filter. Single words are NOT blocked (could be valid)."""
    q = query.lower().strip().rstrip("!.,?")
    if q in CHITCHAT:
        return True
    return any(re.match(p, q) for p in PERSONAL_PATTERNS)


def is_low_quality_result(result: dict) -> bool:
    """Layer 2 — post-pipeline check for no-corpus-match queries."""
    return (
        result.get("coverage_score", 1.0) < 0.25
        and result.get("confidence_score_final", 1.0) < 0.35
    )


# ── Conversation continuity ───────────────────────────────────────────────────
def build_contextual_query(query: str, history: list, n_prior: int = 2) -> str:
    """
    If the current query is very short (<=3 words), prepend the last n_prior
    user messages so the pipeline has enough context to resolve follow-ups.
    e.g. "notre dame" then "campus" → "notre dame campus"
    """
    if len(query.split()) > 3:
        return query
    prior_user = [m["text"] for m in history if m["role"] == "user"]
    if not prior_user:
        return query
    context = " ".join(prior_user[-n_prior:])
    return f"{context} {query}".strip()


# ── HTML helpers ──────────────────────────────────────────────────────────────
def pill(dot_cls: str, text: str) -> str:
    dot = f'<span class="dot {dot_cls}"></span>' if dot_cls else ""
    return f'<span class="pill">{dot}{text}</span>'


def meta_strip_html(meta: dict) -> str:
    c      = meta.get("complexity", "simple")
    model  = meta.get("model_used_final", "—").split("/")[-1]
    ms     = meta.get("total_pipeline_ms", 0)
    conf   = meta.get("confidence_score_final", 0.0)
    saved  = meta.get("compression_tokens_saved", 0)
    k      = meta.get("k_final", meta.get("k", "—"))
    cost   = meta.get("actual_cost", 0.0)
    diag   = meta.get("retrieval_diagnostics") or {}
    lat    = f"{int(ms)}ms" if ms < 1000 else f"{ms/1000:.1f}s"
    cost_s = "free (local)" if cost == 0 else fmt_cost(cost)
    comp_mode = (diag.get("compression_mode") or "standard").replace("_", " ")
    cache_hit = bool(meta.get("cache_hit", False))
    cache_similarity = meta.get("cache_similarity", None)
    pills  = [
        pill(f"dot-{c}", c),
        pill("", model),
        pill("", lat),
        pill("", f"conf {conf:.2f}"),
        pill("", f"{saved} tokens saved"),
        pill("", f"k={k}"),
        pill("", f"mode: {comp_mode}"),
        pill("", cost_s),
    ]
    if cache_hit:
        if isinstance(cache_similarity, (int, float)):
            pills.append(pill("dot-cache", f"cache hit ({float(cache_similarity):.2f})"))
        else:
            pills.append(pill("dot-cache", "cache hit"))
    return f'<div class="meta-strip">{"".join(pills)}</div>'


def render_msg(role: str, text: str, meta: dict = None) -> str:
    if role == "user":
        return f"""<div class="msg-user">
            <div class="bubble bubble-user">{text}</div>
            <div class="avatar avatar-user">You</div>
        </div>"""
    meta_html = meta_strip_html(meta) if meta else ""
    return f"""<div class="msg-bot">
        <div class="avatar avatar-bot">ND</div>
        <div style="display:flex;flex-direction:column">
            <div class="bubble bubble-bot">{text}</div>
            {meta_html}
        </div>
    </div>"""


def badge(c: str) -> str:
    return f'<span class="badge badge-{c}">{c}</span>'


def conf_bar(v: float) -> str:
    pct = int(v * 100)
    return f"""<div class="conf-bar-wrap">
        <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{pct}%"></div></div>
        <span style="font-size:10px;color:#888;min-width:28px;text-align:right">{v:.2f}</span>
    </div>"""


def render_retrieval_diagnostics(diag: dict):
    if not diag:
        st.caption("No retrieval diagnostics available for this turn.")
        return
    cov = diag.get("coverage_score", 0.0)
    st.markdown(
        f"- **k**: `{diag.get('k_base', '—')} → {diag.get('k_final', '—')}`  \n"
        f"- **coverage**: `{cov:.3f}`  \n"
        f"- **hyde**: `attempted={diag.get('hyde_attempted', False)}`, "
        f"`used={diag.get('hyde_used', False)}`, "
        f"`alt_coverage={diag.get('hyde_coverage_score', None)}`  \n"
        f"- **retrieval retry**: `{diag.get('retrieval_retry', False)}`  \n"
        f"- **compression mode**: `{diag.get('compression_mode', 'standard')}`  \n"
        f"- **factual extraction**: `attempted={diag.get('factual_extraction_attempted', False)}`, "
        f"`used={diag.get('factual_extraction_used', False)}`, "
        f"`hits={diag.get('factual_hit_count', 0)}`  \n"
        f"- **grounding gate**: `passed={diag.get('grounding_gate_passed', False)}` "
        f"(`{diag.get('grounding_gate_reason', '')}`)  \n"
        f"- **openai fallback**: `attempted={diag.get('openai_fallback_attempted', False)}`, "
        f"`used={diag.get('openai_fallback_used', False)}`  \n"
        f"- **deep openai retry**: `attempted={diag.get('deep_openai_attempted', False)}`, "
        f"`used={diag.get('deep_openai_used', False)}`"
    )
    top_sources = diag.get("retrieval_sources_top5") or []
    if top_sources:
        lines = []
        for row in top_sources:
            src = row.get("source", "unknown")
            dist = row.get("distance", "")
            lines.append(f"- `{src}` (distance: `{dist}`)")
        st.markdown("**Top retrieved sources**")
        st.markdown("\n".join(lines))
    factual_hits = diag.get("factual_top_hits") or []
    if factual_hits:
        lines = []
        for h in factual_hits[:3]:
            lines.append(f"- `{h.get('score', 0.0)}` · `{h.get('source', 'unknown')}` · {h.get('text', '')}")
        st.markdown("**Top factual sentence hits**")
        st.markdown("\n".join(lines))


# ── Dashboard ─────────────────────────────────────────────────────────────────
def render_dashboard():
    log = st.session_state.query_log
    if not log:
        st.info("No queries yet — ask something in the Chat tab.")
        return

    n           = len(log)
    avg_ms      = sum(r["ms"] for r in log) / n
    avg_conf    = sum(r["conf"] for r in log) / n
    total_saved = sum(r["saved"] for r in log)
    total_in    = sum(r["tokens_in"] for r in log)
    total_out   = sum(r["tokens_out"] for r in log)
    simple_n    = sum(1 for r in log if r["complexity"] == "simple")
    medium_n    = sum(1 for r in log if r["complexity"] == "medium")
    complex_n   = sum(1 for r in log if r["complexity"] == "complex")
    local_pct   = round((simple_n + medium_n) / n * 100) if n else 0
    retry_rate  = round((sum(1 for r in log if r.get("retrieval_retry")) / n) * 100) if n else 0
    factual_used_n = sum(1 for r in log if r.get("factual_extraction_used"))
    openai_fb_n = sum(1 for r in log if r.get("openai_fallback_used"))
    grounded_n = sum(1 for r in log if r.get("grounding_gate_passed"))

    # Pricing calculations
    actual_total   = sum(r["actual_cost"] for r in log)
    baseline_total = sum(r["baseline_cost"] for r in log)
    dollar_saved   = baseline_total - actual_total
    pct_saved      = round(dollar_saved / baseline_total * 100) if baseline_total > 0 else 0
    compression_savings = sum(
        calc_baseline_cost(r["original_tokens"], r["tokens_out"]) -
        calc_baseline_cost(r["tokens_in"], r["tokens_out"])
        for r in log
    )
    routing_savings = sum(
        calc_baseline_cost(r["tokens_in"], r["tokens_out"]) - r["actual_cost"]
        for r in log
    )

    lat_fmt = f"{int(avg_ms)}ms" if avg_ms < 1000 else f"{avg_ms/1000:.1f}s"

    # ── Pricing section ───────────────────────────────────────────────────────
    st.markdown('<div class="section-title">💰 cost analysis</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="price-grid">
      <div class="price-card price-card-spend">
        <div class="price-label">Actual spend</div>
        <div class="price-val">{fmt_cost(actual_total)}</div>
        <div class="price-sub">with routing + compression</div>
      </div>
      <div class="price-card price-card-baseline">
        <div class="price-label">Baseline (naive)</div>
        <div class="price-val" style="color:#92400E">{fmt_cost(baseline_total)}</div>
        <div class="price-sub">all queries → gpt-4o-mini, no compression</div>
      </div>
      <div class="price-card price-card-saved">
        <div class="price-label">Money saved</div>
        <div class="price-val-saved">{fmt_cost(dollar_saved)}</div>
        <div class="price-sub">routing + compression combined</div>
      </div>
      <div class="price-card price-card-pct">
        <div class="price-label-inv">Cost reduction</div>
        <div class="price-val-inv">{pct_saved}%</div>
        <div class="price-sub-inv">vs unoptimized pipeline</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Retrieval diagnostics summary ─────────────────────────────────────────
    st.markdown('<div class="section-title">retrieval diagnostics</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="stat-grid">
      <div class="stat-card"><div class="stat-label">Avg k final</div><div class="stat-val">{sum(r.get("k", 0) for r in log)/n:.1f}</div><div class="stat-sub">post-retry retrieval depth</div></div>
      <div class="stat-card"><div class="stat-label">Avg coverage</div><div class="stat-val">{sum(r.get("coverage", 0.0) for r in log)/n:.2f}</div><div class="stat-sub">query-term coverage in retrieved docs</div></div>
      <div class="stat-card"><div class="stat-label">Retrieval retry rate</div><div class="stat-val">{retry_rate}%</div><div class="stat-sub">queries that triggered robust retry</div></div>
      <div class="stat-card"><div class="stat-label">Factual extractive used</div><div class="stat-val">{factual_used_n}</div><div class="stat-sub">turns answered via sentence extraction</div></div>
      <div class="stat-card"><div class="stat-label">Grounded responses</div><div class="stat-val">{grounded_n}/{n}</div><div class="stat-sub">grounding gate passed</div></div>
      <div class="stat-card"><div class="stat-label">OpenAI fallback used</div><div class="stat-val">{openai_fb_n}</div><div class="stat-sub">fallback accepted after checks</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="price-breakdown">
      <div style="font-size:11px;font-weight:600;color:#888;text-transform:uppercase;letter-spacing:.05em;margin-bottom:8px">
        Savings breakdown
      </div>
      <div class="price-row">
        <span>Saved by routing simple/medium to local models</span>
        <span class="price-row-green">{fmt_cost(routing_savings)}</span>
      </div>
      <div class="price-row">
        <span>Saved by context compression (fewer input tokens)</span>
        <span class="price-row-green">{fmt_cost(compression_savings)}</span>
      </div>
      <div class="price-row">
        <span>Tokens removed by compression</span>
        <span class="price-row-green">{total_saved:,} tokens</span>
      </div>
      <div class="price-row">
        <span style="font-weight:600">Total saved vs naive GPT-4o-mini</span>
        <span class="price-row-green" style="font-weight:600">{fmt_cost(dollar_saved)}</span>
      </div>
    </div>
    <div style="font-size:10px;color:#aaa;margin-bottom:16px">
      Rates: gpt-4o-mini $0.150/M input · $0.600/M output · Local models $0.000 (free).
      Baseline assumes all queries sent to gpt-4o-mini at original (pre-compression) token count.
    </div>
    """, unsafe_allow_html=True)

    # ── Session summary ───────────────────────────────────────────────────────
    st.markdown('<div class="section-title">session summary</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="stat-grid">
      <div class="stat-card"><div class="stat-label">Avg latency</div><div class="stat-val">{lat_fmt}</div><div class="stat-sub">end-to-end per query</div></div>
      <div class="stat-card"><div class="stat-label">Avg confidence</div><div class="stat-val">{avg_conf:.2f}</div><div class="stat-sub">answer grounding score</div></div>
      <div class="stat-card"><div class="stat-label">Tokens saved</div><div class="stat-val">{total_saved:,}</div><div class="stat-sub">by context compression</div></div>
      <div class="stat-card"><div class="stat-label">Prompt tokens used</div><div class="stat-val">{total_in:,}</div><div class="stat-sub">total input sent to LLM</div></div>
      <div class="stat-card"><div class="stat-label">Completion tokens</div><div class="stat-val">{total_out:,}</div><div class="stat-sub">total output generated</div></div>
      <div class="stat-card"><div class="stat-label">Free queries</div><div class="stat-val">{local_pct}%</div><div class="stat-sub">{simple_n + medium_n} of {n} ran locally</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Model routing ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">model routing</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="route-grid">
      <div class="route-card"><div class="route-count">{simple_n}</div><div class="route-label">simple</div><div class="route-model">tinyllama · local · free</div></div>
      <div class="route-card"><div class="route-count">{medium_n}</div><div class="route-label">medium</div><div class="route-model">mistral-3b · local · free</div></div>
      <div class="route-card"><div class="route-count">{complex_n}</div><div class="route-label">complex</div><div class="route-model">gpt-4o-mini · api · paid</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Recent queries table ──────────────────────────────────────────────────
    st.markdown('<div class="section-title">recent queries</div>', unsafe_allow_html=True)
    rows = ""
    for r in reversed(log):
        ms_fmt     = f"{int(r['ms'])}ms" if r["ms"] < 1000 else f"{r['ms']/1000:.1f}s"
        model_s    = r["model"].split("/")[-1]
        q_trunc    = r["query"][:52] + "…" if len(r["query"]) > 52 else r["query"]
        cost_s     = "free" if r["actual_cost"] == 0 else fmt_cost(r["actual_cost"])
        saved_s    = fmt_cost(r["baseline_cost"] - r["actual_cost"])
        rows += f"""<tr>
            <td title="{r['query']}">{q_trunc}</td>
            <td>{badge(r['complexity'])}</td>
            <td><span class="model-chip">{model_s}</span></td>
            <td>{conf_bar(r['conf'])}</td>
            <td style="color:#888">{ms_fmt}</td>
            <td style="color:#888;font-family:'DM Mono',monospace">{cost_s}</td>
            <td style="color:#15803D;font-family:'DM Mono',monospace">{saved_s}</td>
        </tr>"""

    st.markdown(f"""
    <div style="overflow-x:auto">
    <table class="q-table">
      <colgroup>
        <col style="width:28%"><col style="width:11%"><col style="width:16%">
        <col style="width:16%"><col style="width:8%"><col style="width:11%"><col style="width:10%">
      </colgroup>
      <thead><tr>
        <th>query</th><th>complexity</th><th>model</th>
        <th>confidence</th><th>latency</th><th>cost</th><th>saved</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)


# ── App layout ────────────────────────────────────────────────────────────────
header_placeholder = st.empty()
with header_placeholder.container():
    _render_header(st.session_state.get("active_session_title", "AI Assistant"))

chat_tab, dash_tab = st.tabs(["💬  Chat", "📊  Dashboard"])


# ── CHAT TAB ──────────────────────────────────────────────────────────────────
with chat_tab:

    # Load pipeline whenever unavailable (prevents stale "stuck" startup errors).
    if st.session_state.pipeline is None:
        with st.spinner("Loading pipeline and vector database…"):
            pipeline, err = load_pipeline()
            st.session_state.pipeline = pipeline
            st.session_state.pipeline_error = err

    if st.session_state.pipeline_error:
        st.error(f"Pipeline failed to load: {st.session_state.pipeline_error}")
        st.info("Check that LM Studio is running and your .env is configured.")
        if st.button("Retry pipeline load"):
            load_pipeline.clear()
            st.session_state.pipeline = None
            st.session_state.pipeline_error = None
            st.rerun()
        st.stop()

    uploaded_files = st.file_uploader(
        "Upload file (PDF/TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Uploaded files are used as session context and are not permanently indexed.",
    )
    session_documents = []
    session_uploads = []
    if uploaded_files:
        skipped_files = 0
        for f in uploaded_files:
            payload = _extract_session_upload(f)
            extracted = payload.get("text", "")
            session_uploads.append(payload)
            if len(extracted) >= 80:
                session_documents.append(extracted)
            else:
                skipped_files += 1
        first_name = Path(uploaded_files[0].name or "").stem if uploaded_files else ""
        st.session_state.active_session_title = first_name or "AI Assistant"
        st.caption(
            f"Session docs loaded: {len(session_documents)} file(s)"
            + (f" · skipped {skipped_files} file(s) with too little extractable text" if skipped_files else "")
        )
        st.radio(
            "Session retrieval mode",
            options=["upload_only", "hybrid"],
            format_func=lambda x: "Upload only (ignore base corpus)" if x == "upload_only" else "Hybrid (uploaded files + base corpus)",
            horizontal=True,
            key="session_retrieval_mode",
            help="Choose whether uploaded files replace corpus retrieval or augment it.",
        )
    else:
        st.session_state.active_session_title = "AI Assistant"

    with header_placeholder.container():
        _render_header(st.session_state.active_session_title)

    # Render history
    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align:center;padding:40px 0;color:#aaa">
          <div style="font-size:36px;margin-bottom:10px">🤖</div>
          <div style="font-size:15px;font-weight:500;color:#666;margin-bottom:8px">
            Hi there! Ask me anything and I will try to help.
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        parts = ['<div class="chat-wrap">']
        for msg in st.session_state.messages:
            # off-topic messages stored with a flag — render separately
            if msg.get("off_topic"):
                continue
            parts.append(render_msg(msg["role"], msg["text"], msg.get("meta")))
        parts.append("</div>")
        st.markdown("".join(parts), unsafe_allow_html=True)

        # Render off-topic warnings cleanly using st.warning (no HTML leak)
        for msg in st.session_state.messages:
            if msg.get("off_topic"):
                st.warning(msg["text"])

        # Retrieval diagnostics (per assistant turn)
        assistant_turn = 0
        for msg in st.session_state.messages:
            if msg.get("role") != "bot" or msg.get("off_topic"):
                continue
            assistant_turn += 1
            diag = (msg.get("meta") or {}).get("retrieval_diagnostics") or {}
            if not diag:
                continue
            with st.expander(f"Retrieval diagnostics · turn {assistant_turn}", expanded=False):
                render_retrieval_diagnostics(diag)

    # Input form
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    with st.form("chat_form", clear_on_submit=True):
        col_in, col_btn = st.columns([10, 1])
        with col_in:
            user_input = st.text_input(
                "query",
                placeholder="Ask a question or upload a file for context…",
                label_visibility="collapsed",
            )
        with col_btn:
            submitted = st.form_submit_button("→", use_container_width=True)

    if submitted and user_input.strip():
        query = user_input.strip()
        st.session_state.messages.append({"role": "user", "text": query})

        # Layer 1 — pattern filter
        if is_off_topic(query):
            st.session_state.messages.append({
                "role": "bot",
                "text": (
                    "I can only answer based on the available context and uploaded files. "
                    "Try rephrasing your question or upload a relevant document."
                ),
                "off_topic": True,
            })
            st.rerun()

        # Build contextual query for short follow-ups only when no uploads are active.
        # Uploaded documents already provide the working context; stitching prior turns
        # can contaminate answers across unrelated questions.
        if session_uploads:
            contextual_query = query
        else:
            prior = st.session_state.messages[:-1]  # exclude the one just appended
            contextual_query = build_contextual_query(query, prior, n_prior=2)

        # Run pipeline
        with st.spinner("Retrieving and generating answer…"):
            t0     = time.perf_counter()
            result = st.session_state.pipeline.run(
                contextual_query,
                session_documents=session_documents if session_documents else None,
                session_uploads=session_uploads if session_uploads else None,
                session_retrieval_mode=st.session_state.get("session_retrieval_mode", "upload_only"),
            )
            ui_ms  = round((time.perf_counter() - t0) * 1000, 1)

        # Layer 2 — low coverage + low confidence
        if is_low_quality_result(result):
            st.session_state.messages.append({
                "role": "bot",
                "text": (
                    "I couldn't find relevant information in the current context. "
                    "Please rephrase the question or upload a document with the needed details."
                ),
                "off_topic": True,
            })
            st.rerun()

        # Compute costs
        complexity     = result.get("complexity", "simple")
        is_local       = not is_openai_billed_result(result)
        tokens_in      = result.get("input_tokens", 0)
        tokens_out     = result.get("output_tokens", 0)
        original_tok   = result.get("original_token_count", tokens_in)
        actual_cost    = calc_cost(tokens_in, tokens_out, is_local)
        baseline_cost  = calc_baseline_cost(original_tok, tokens_out)

        meta = {
            "complexity":               complexity,
            "model_used_final":         result.get("model_used_final", "—"),
            "total_pipeline_ms":        result.get("total_pipeline_ms", ui_ms),
            "confidence_score_final":   result.get("confidence_score_final", 0.0),
            "compression_tokens_saved": result.get("compression_tokens_saved", 0),
            "k_final":                  result.get("k_final", result.get("k", "—")),
            "actual_cost":              actual_cost,
            "cache_hit":                result.get("cache_hit", False),
            "cache_similarity":         result.get("cache_similarity"),
            "retrieval_diagnostics":    result.get("retrieval_diagnostics", {}),
        }

        answer = result.get("final_answer", "No answer generated.")
        st.session_state.messages.append({"role": "bot", "text": answer, "meta": meta})

        st.session_state.query_log.append({
            "query":           query,
            "complexity":      complexity,
            "model":           result.get("model_used_final", "—"),
            "ms":              result.get("total_pipeline_ms", ui_ms),
            "conf":            result.get("confidence_score_final", 0.0),
            "tokens_in":       tokens_in,
            "tokens_out":      tokens_out,
            "original_tokens": original_tok,
            "saved":           result.get("compression_tokens_saved", 0),
            "k":               result.get("k_final", result.get("k", "—")),
            "coverage":        result.get("coverage_score", 0.0),
            "retried":         result.get("retried", False),
            "cache_hit":       result.get("cache_hit", False),
            "retrieval_retry": result.get("retrieval_retry", False),
            "factual_extraction_used": (result.get("retrieval_diagnostics") or {}).get("factual_extraction_used", False),
            "openai_fallback_used": result.get("openai_fallback_used", False),
            "grounding_gate_passed": result.get("grounding_gate_passed", False),
            "actual_cost":     actual_cost,
            "baseline_cost":   baseline_cost,
        })
        st.rerun()


# ── DASHBOARD TAB ─────────────────────────────────────────────────────────────
with dash_tab:
    render_dashboard()
