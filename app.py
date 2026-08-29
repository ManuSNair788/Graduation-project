import json
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Streamlit Community Cloud (and HF Spaces, depending on config) inject secrets via
# st.secrets, not the process environment — llm.py only ever reads os.environ (it's a plain
# module, not Streamlit-aware, and is shared with the CLI pipeline scripts where st.secrets
# doesn't exist at all). Bridge the two here, before any pipeline module is imported, so a
# plain os.environ lookup in llm.py still works under either hosting platform. Wrapped in
# try/except because st.secrets raises if no secrets are configured at all (e.g. local dev
# relying solely on discovery-engine/.env, which llm.py's own _load_dotenv() already handles).
try:
    for _key, _value in st.secrets.items():
        os.environ.setdefault(_key, str(_value))
except Exception:
    pass

from pipeline import stage1_filter, stage2_extract, stage3_aggregate

st.set_page_config(page_title="AJIO Discovery Engine", layout="wide")

DATA_DIR = Path("data")


@st.cache_data
def load_cached():
    def _load(name):
        path = DATA_DIR / name
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    return {
        "raw": _load("raw_snippets.json"),
        "filtered": _load("filtered_snippets.json"),
        "extractions": _load("extractions.json"),
        "aggregates": _load("aggregates.json"),
        "synthesis": _load("synthesis.json"),
        "stage1_report": _load("stage1_report.json"),
    }


def render_opportunity_table(aggregates: dict, key_prefix: str = "") -> None:
    if not aggregates or not aggregates.get("barrier_table"):
        st.info("No aggregate data available.")
        return

    total_records = aggregates.get("total_records")
    non_other_total = aggregates.get("non_other_total")
    ranked_total = aggregates.get("ranked_total", non_other_total)
    other = aggregates.get("other_summary")
    insufficient = aggregates.get("insufficient_evidence_summary")

    excluded_bits = []
    if other and other["count"]:
        excluded_bits.append(f"{other['count']} classified as `other`")
    if insufficient and insufficient["total_count"]:
        excluded_bits.append(f"{insufficient['total_count']} below the evidence threshold")
    if excluded_bits:
        st.caption(
            f"Ranked below: **{ranked_total} of {total_records}** classified snippets "
            f"({' and '.join(excluded_bits)} are excluded from ranking and reported below the table)."
        )

    df = pd.DataFrame(aggregates["barrier_table"]).sort_values("opportunity_score", ascending=False)
    st.dataframe(df, width="stretch", hide_index=True)
    st.bar_chart(df.set_index("barrier")["opportunity_score"], horizontal=True)
    st.caption(
        "**Opportunity Score = frequency_percent x mean_intensity x addressability_weight** "
        "(addressability_weight = 1.0 if the majority of that barrier's snippets are "
        f"addressable_without_money, else 0.3). frequency_percent for the {len(df)} ranked "
        f"barriers is computed against **{ranked_total}** (total minus `other` minus the "
        f"below-threshold categories), not the full **{total_records}**."
    )

    if other and other["count"]:
        st.warning(
            f"**`other` — {other['count']} snippets ({other['percent_of_corpus']}% of the "
            f"{total_records}-snippet corpus) — excluded from the ranking above.**\n\n{other['note']}"
        )

    if insufficient and insufficient["items"]:
        rows = "\n".join(f"- `{item['barrier']}` — {item['count']} snippet(s)" for item in insufficient["items"])
        st.warning(
            f"**Insufficient evidence to rank** (fewer than {insufficient['min_evidence_count']} "
            f"snippets):\n\n{rows}\n\n{insufficient['note']}"
        )


def render_corpus_provenance(stage1_report: dict) -> None:
    if not stage1_report:
        st.info("No Stage 1 provenance data available.")
        return

    collected = stage1_report["input_count"]
    presampled = stage1_report["presampled_count"]
    relevant = stage1_report["survivor_count"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Collected (raw)", collected)
    c2.metric("Presampled / filtered", presampled)
    c3.metric("Relevant (Stage 1 output)", relevant)

    by_source = stage1_report.get("by_source", {})
    rows = [
        {"source": source, "survived": v["survived"], "of": v["of"], "rate": f"{v['rate']:.1%}"}
        for source, v in by_source.items()
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption(
        f"{collected} snippets scraped -> presampled down to {presampled} before Stage 1 "
        f"(weighted by source, see ARCHITECTURE.md §2.3) -> {relevant} survived the relevance "
        f"filter and went into Stage 2 extraction."
    )


def run_live_pipeline(pasted_text: str) -> dict:
    lines = [line.strip() for line in pasted_text.splitlines() if line.strip()]
    snippets = [
        {"id": f"pasted_{i:04d}", "source": "pasted", "app": None, "date": None, "rating": None, "text": line}
        for i, line in enumerate(lines, start=1)
    ]

    survivors, _ = stage1_filter.classify_snippets(snippets, checkpoint_path=None)
    records, dropped = stage2_extract.extract_snippets(survivors)
    aggregates = stage3_aggregate.aggregate_records(records) if records else None

    return {
        "input_count": len(snippets),
        "survivor_count": len(survivors),
        "extracted_count": len(records),
        "dropped_count": len(dropped),
        "aggregates": aggregates,
        "records": records,
    }


st.title("AJIO Discovery Engine")
st.caption("Ranked, scored purchase-barrier analysis from public user feedback - not a summary.")

tab1, tab2, tab3, tab4 = st.tabs(["Run", "Opportunity table", "Evidence", "Questions"])

with tab1:
    st.subheader("Cached corpus results")
    st.write(
        "Loads the pre-computed pipeline output for the bundled corpus - no live model calls, "
        "renders instantly."
    )
    if st.button("Load cached results", type="primary"):
        cached = load_cached()
        st.session_state["cached"] = cached

    if "cached" in st.session_state:
        cached = st.session_state["cached"]
        if cached["aggregates"]:
            n = cached["aggregates"]["total_records"]
            st.success(f"Loaded - {n} classified snippets from the bundled corpus.")
            render_opportunity_table(cached["aggregates"], key_prefix="cached")
        else:
            st.warning("No cached aggregates found - run the pipeline first.")

    st.divider()
    st.subheader("Try it yourself")
    st.write(
        "Paste your own reviews below (one per line) and run Stages 1-3 live against them. "
        "This does not touch the cached corpus results above."
    )
    pasted = st.text_area(
        "One review per line",
        height=150,
        placeholder=(
            "Nice fabric but not sure about the size\n"
            "Saved this for a wedding, need to check if it's formal enough\n"
            "Too many similar kurtas in my wishlist, can't decide"
        ),
    )
    if st.button("Analyze my reviews"):
        if not pasted.strip():
            st.warning("Paste at least one review first.")
        else:
            with st.spinner("Running Stages 1-3 live..."):
                result = run_live_pipeline(pasted)
            st.success(
                f"{result['input_count']} pasted -> {result['survivor_count']} relevant -> "
                f"{result['extracted_count']} extracted ({result['dropped_count']} dropped)"
            )
            if result["aggregates"]:
                render_opportunity_table(result["aggregates"], key_prefix="live")
            else:
                st.info("None of the pasted reviews were classified as relevant to wishlisting/deferral.")

with tab2:
    st.subheader("Corpus provenance")
    st.write(
        "Sample size and where it came from — visible up front, since presenting the ranking "
        "without this context would make it easy to over-read a small, filtered sample."
    )
    cached = load_cached()
    render_corpus_provenance(cached["stage1_report"])

    st.divider()
    st.subheader("Opportunity table")
    render_opportunity_table(cached["aggregates"])

with tab3:
    st.subheader("Evidence")
    cached = load_cached()
    extractions = cached["extractions"]
    if not extractions:
        st.info("No extraction data available.")
    else:
        barriers = sorted({r["barrier"] for r in extractions})
        selected = st.selectbox("Barrier", barriers)
        rows = [r for r in extractions if r["barrier"] == selected]
        df = pd.DataFrame(rows)[["source", "evidence", "workaround", "intensity", "addressable_without_money"]]
        st.dataframe(df, width="stretch", hide_index=True)

with tab4:
    st.subheader("The ten questions")
    cached = load_cached()
    extractions = cached["extractions"] or []
    aggregates = cached["aggregates"] or {}
    barrier_table = aggregates.get("barrier_table", [])
    insufficient = aggregates.get("insufficient_evidence_summary", {"items": [], "total_count": 0})
    synthesis = cached["synthesis"] or {}

    def barrier_row(name):
        return next((b for b in barrier_table if b["barrier"] == name), None)

    def insufficient_row(name):
        return next((b for b in insufficient.get("items", []) if b["barrier"] == name), None)

    total_extractions = len(extractions)
    save_intent_counts = Counter(r["save_intent"] for r in extractions)
    unclear_count = save_intent_counts.get("unclear", 0)
    classifiable_count = total_extractions - unclear_count
    classifiable_counts = Counter(
        r["save_intent"] for r in extractions if r["save_intent"] != "unclear"
    )
    journey_saved_revisit = [r for r in extractions if r["journey_stage"] in ("saved", "revisit")]
    journey_barrier_counts = Counter(r["barrier"] for r in journey_saved_revisit)
    info_sought = [r["info_sought_outside_app"] for r in extractions if r.get("info_sought_outside_app")]
    workarounds = [r["workaround"] for r in extractions if r.get("workaround")]

    st.markdown("**1. Why do users add fashion products to their wishlist?**")
    st.write(
        f"`save_intent` was unclear for **{unclear_count} of {total_extractions}** classified "
        f"snippets — too little context in the snippet itself to tell. The answer below comes "
        f"from the **{classifiable_count}** that were classifiable."
    )
    st.dataframe(
        pd.DataFrame(
            [{"save_intent": k, "count": v} for k, v in classifiable_counts.most_common()]
        ),
        width="stretch",
        hide_index=True,
    )

    st.markdown("**2. What prevents wishlisted products from eventually being purchased?**")
    top3 = barrier_table[:3]
    if top3:
        lead = ", ".join(f"`{b['barrier']}`" for b in top3)
        st.write(
            f"The top three ranked barriers are {lead}, led by `{top3[0]['barrier']}` at an "
            f"Opportunity Score of **{top3[0]['opportunity_score']}** ({top3[0]['count']} snippets)."
        )
    st.dataframe(
        pd.DataFrame([{"barrier": b["barrier"], "opportunity_score": b["opportunity_score"], "count": b["count"]} for b in barrier_table]),
        width="stretch",
        hide_index=True,
    )

    st.markdown("**3. What uncertainties remain after users identify a product they like?**")
    st.write(
        f"Restricting to snippets where `journey_stage` is saved or revisit "
        f"(**{len(journey_saved_revisit)}** snippets) surfaces which barriers persist after the "
        f"product is already chosen:"
    )
    st.dataframe(
        pd.DataFrame(
            [{"barrier": k, "count": v} for k, v in sorted(journey_barrier_counts.items(), key=lambda kv: -kv[1])]
        ),
        width="stretch",
        hide_index=True,
    )

    st.markdown("**4. What causes users to postpone a purchase?**")
    by_intensity = sorted(barrier_table, key=lambda b: -b["mean_intensity"])[:3]
    if by_intensity:
        st.write(
            f"`{by_intensity[0]['barrier']}` carries the highest mean intensity at "
            f"**{by_intensity[0]['mean_intensity']}** (scale per ARCHITECTURE.md §2.5), "
            f"followed by " + ", ".join(f"`{b['barrier']}` ({b['mean_intensity']})" for b in by_intensity[1:]) + "."
        )
    st.dataframe(
        pd.DataFrame([{"barrier": b["barrier"], "mean_intensity": b["mean_intensity"]} for b in by_intensity]),
        width="stretch",
        hide_index=True,
    )

    st.markdown("**5. How do users compare multiple shortlisted products?**")
    co = insufficient_row("choice_overload")
    if co:
        st.write(
            f"`choice_overload` has only **{co['count']} snippet** in the corpus — below the "
            f"n>=3 evidence threshold (Tab 2), so it is not part of the ranked barrier table. "
            f"The single snippet behind it (\"please pin the link of white top\") is a link "
            f"request, not evidence of comparison-driven hesitation, which is why it does not "
            f"support a ranked score."
        )
    else:
        st.write("`choice_overload` did not appear in this run's extractions.")

    st.markdown("**6. What information do users seek outside AJIO before purchasing?**")
    st.write(
        f"**{len(info_sought)}** of {total_extractions} snippets named something the user sought "
        f"outside the app before deciding. A sample:"
    )
    st.dataframe(pd.DataFrame({"info_sought_outside_app": info_sought[:5]}), width="stretch", hide_index=True)

    st.markdown("**7. What role do fit, size, styling, price, reviews, occasion, social validation play?**")
    st.write(
        "The full ranked barrier table (Tab 2) answers this directly — each row's "
        "`frequency_percent` is that theme's share of the classifiable, above-threshold corpus:"
    )
    st.dataframe(
        pd.DataFrame([{"barrier": b["barrier"], "count": b["count"], "frequency_percent": b["frequency_percent"]} for b in barrier_table]),
        width="stretch",
        hide_index=True,
    )

    st.markdown("**8. When is the wishlist genuine purchase intent vs simply a bookmark?**")
    st.write(
        "Cross-cutting each barrier by `save_intent` shows which ones skew toward "
        "`purchase_intent` versus `bookmarking` or `aspiration`:"
    )
    cross_si = aggregates.get("barrier_x_save_intent", {})
    si_rows = []
    for barrier, counts in cross_si.items():
        row = {"barrier": barrier}
        row.update(counts)
        si_rows.append(row)
    st.dataframe(pd.DataFrame(si_rows), width="stretch", hide_index=True)

    st.markdown("**9. How do these behaviours differ across user segments?**")
    st.write(
        "`segment_signal` is free-text per snippet (see ARCHITECTURE.md §2.5), so most values "
        "are unique to a single snippet rather than a reusable label. For the ranked barriers, "
        "the table below shows how many distinct segment descriptions appear per barrier and "
        "the one description repeated most often, if any repeated at all:"
    )
    cross_seg = aggregates.get("barrier_x_segment_signal", {})
    seg_rows = []
    for b in barrier_table:
        barrier = b["barrier"]
        counts = cross_seg.get(barrier, {})
        top_label, top_count = max(counts.items(), key=lambda kv: kv[1]) if counts else (None, 0)
        seg_rows.append(
            {
                "barrier": barrier,
                "distinct_segment_descriptions": len(counts),
                "most_repeated": top_label if top_count > 1 else "(none repeated)",
                "repeat_count": top_count if top_count > 1 else 0,
            }
        )
    st.dataframe(pd.DataFrame(seg_rows), width="stretch", hide_index=True)

    st.markdown("**10. What unmet needs emerge consistently across user conversations?**")
    st.write(f"**{len(workarounds)}** of {total_extractions} snippets described a workaround the user improvised. A sample:")
    st.dataframe(pd.DataFrame({"workaround": workarounds[:5]}), width="stretch", hide_index=True)
    if synthesis.get("available"):
        st.write("Stage 4 synthesis, generated from the full corpus:")
        st.markdown(synthesis["text"])
    else:
        st.info("Stage 4 synthesis unavailable for this run.")
